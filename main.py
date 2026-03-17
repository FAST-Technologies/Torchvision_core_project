# # main.py

# # Импорт основных библиотек
# from SegmentationTester import SegmentationTester
# from TorchSegmenter import TorchSegmenter
# from NeuralSegmenter import NeuralSegmenter
# from cv2SklearnSegmenter import CV2SklearnSegmenter
# from segmentation_metrics import SegmentationMetrics
# from SegmentationComparator import SegmentationComparator
# from tester import ExtendedSegmentationComparator
# from HelpModule import create_advanced_pipeline_example, analyze_pipeline_results, print_pipeline_analysis, compare_segmentation_methods_with_timing, original_compare_segmentation_methods
# import pandas as pd
# from typing import Union, Dict, Any, Tuple, Optional, List
# from huggingface_hub import hf_hub_download
# import matplotlib.pyplot as plt
# import numpy as np
# import requests
# from io import BytesIO
# from PIL import Image
# import os
# import shutil
# import datetime
# import traceback
# import warnings

# def main() -> Union[SegmentationTester, Dict[str, Any], pd.DataFrame]:
#     """Тестирование использования всех классов"""
    
#     # Инициализация тестера
#     tester = SegmentationTester()
    
#     # ============ ДОБАВЛЕНИЕ МЕТОДОВ ============
    
#     print("=" * 60)
#     print("ИНИЦИАЛИЗАЦИЯ МЕТОДОВ СЕГМЕНТАЦИИ")
#     print("=" * 60)
    
#     # 1. Методы CV2/Sklearn
#     print("\n1. Загрузка методов CV2/Sklearn...")
#     cv2_methods = {
#         "Global_Threshold_CV2": CV2SklearnSegmenter("global_thresholding", threshold=127),
#         "Adaptive_Threshold_CV2": CV2SklearnSegmenter("adaptive_thresholding", block_size=11, C=2),
#         "Otsu_CV2": CV2SklearnSegmenter("otsu_thresholding"),
#         "Region_Growing_CV2": CV2SklearnSegmenter("region_growing", seed=(100, 100), tolerance=20),
#         "Split_and_Merge_CV2": CV2SklearnSegmenter("split_and_merge", min_region_size=50, threshold=20),
#         "Sobel_CV2": CV2SklearnSegmenter("sobel_edge", threshold=50),
#         "Canny_CV2": CV2SklearnSegmenter("canny_edge", low=50, high=150),
#         "KMeans_CV2": CV2SklearnSegmenter("kmeans_segmentation", k=3),
#         # "DBSCAN_CV2": CV2SklearnSegmenter("dbscan_segmentation", eps=10, min_samples=100),
#         "Active_Contour_CV2": CV2SklearnSegmenter("active_contour", alpha=0.01, beta=0.1, gamma=0.001, max_iterations=2000),
#         "GVF_Contour_CV2": CV2SklearnSegmenter("gvf_contour", mu=0.2, iterations=100),
#         "Watershed_CV2": CV2SklearnSegmenter("watershed"),
#         # "Meanshift_CV2": CV2SklearnSegmenter("meanshift", bandwidth=0.5),
#         "GrabCut_CV2": CV2SklearnSegmenter("grabcut", iterations=10),
#         "FloodFill_CV2": CV2SklearnSegmenter("floodfill", seed=(100, 100), tolerance=20),
#         "Morphological_Snakes_CV2": CV2SklearnSegmenter("morphological_snakes", iterations=100, smoothing=1, threshold=0.5),
#         # "Quickshift_CV2": CV2SklearnSegmenter("quickshift", bandwidth=0.5),
#         "Slic_CV2": CV2SklearnSegmenter("slic", n_segments=100, compactness=10.0),
#         "Felzenszwalb_CV2": CV2SklearnSegmenter("felzenszwalb", scale=100, sigma=0.8, min_size=50),
#         "Chan_Vese_CV2": CV2SklearnSegmenter("chan_vese", mu=0.25, lambda1=1.0, lambda2=1.0, tol=1e-3, max_iter=100),
#         "Threshold_Niblack_CV2": CV2SklearnSegmenter("threshold_niblack", window_size=15, k=0.2),
#         "Threshold_Sauvola_CV2": CV2SklearnSegmenter("threshold_sauvola", window_size=15, k=0.2, r=128),
#         "Random_Walker_CV2": CV2SklearnSegmenter("random_walker", scale=100, sigma=0.8, min_size=50),
#     }
    
#     for name, segmenter in cv2_methods.items():
#         tester.add_method(name, segmenter)
#         print(f"   ✅ {name}")
    
#     # 2. Методы PyTorch
#     print("\n2. Загрузка методов PyTorch...")
#     torch_methods = {
#         "Global_Threshold_Torch": TorchSegmenter("global_thresholding", threshold=0.5),
#         "Adaptive_Threshold_Torch": TorchSegmenter("adaptive_thresholding", block_size=11, C=2),
#         "Otsu_Torch": TorchSegmenter("otsu_thresholding"),
#         # "Region_Growing_Torch": TorchSegmenter("region_growing", seed=(100, 100), tolerance=0.1),
#         "Split_and_Merge_Torch": TorchSegmenter("split_and_merge", min_size=50),
#         "Sobel_Torch": TorchSegmenter("sobel_edge", threshold=0.1),
#         "Canny_Torch": TorchSegmenter("canny_edge", low=0.1, high=0.3),
#         "KMeans_Torch": TorchSegmenter("kmeans_segmentation", k=3),
#         # "DBSCAN_Torch": TorchSegmenter("dbscan_segmentation", eps=10, min_samples=100),
#         # "Active_Contour_Torch": TorchSegmenter("active_contour"),
#         # "GVF_Contour_Torch": TorchSegmenter("gvf_contour"),
#         "Watershed_Torch": TorchSegmenter("watershed"),
#         # "Meanshift_Torch": TorchSegmenter("meanshift", bandwidth=0.5, spatial_radius=35, color_radius=60),
#         "Grabcut_Torch": TorchSegmenter("grabcut", rect=(100, 100, 200, 200), num_iterations=5),
#         "FloodFill_Torch": TorchSegmenter("floodfill", tolerance=0.15),
#         #     "Morphological_Snakes_Torch": TorchSegmenter("morphological_snakes", iterations=100, smoothing=1, threshold=0.5),
#         #     # "Quickshift_Torch": TorchSegmenter("quickshift", bandwidth=0.5),
#         #     "Slic_Torch": TorchSegmenter("slic", n_segments=100, compactness=10.0),
#         #     "Felzenszwalb_Torch": TorchSegmenter("felzenszwalb", scale=100, sigma=0.8, min_size=50),
#         #     "Chan_Vese_Torch": TorchSegmenter("chan_vese", mu=0.25, lambda1=1.0, lambda2=1.0, tol=1e-3, max_iter=100),
#         #     "Threshold_Niblack_Torch": TorchSegmenter("threshold_niblack", window_size=15, k=0.2),
#         #     "Threshold_Sauvola_Torch": TorchSegmenter("threshold_sauvola", window_size=15, k=0.2, r=128),
#         #     "Random_Walker_Torch": TorchSegmenter("random_walker", scale=100, sigma=0.8, min_size=50),
#     }
    
#     for name, segmenter in torch_methods.items():
#         tester.add_method(name, segmenter)
#         print(f"   ✅ {name}")

#     # 3. Нейросетевая сегментация
#     print("\n3. Загрузка нейросетевых методов...")
#     try:
#         # Используйте локальный путь к модели
#         neural_segmenter = NeuralSegmenter(
#             local_path="/home/yamshchikov/models/segformer-b5-ready"
#         )
#         tester.add_method("Neural_SegFormer", neural_segmenter)
#         print(f"   ✅ Neural_SegFormer")
#     except Exception as e:
#         print(f"   ❌ Neural_SegFormer - ошибка: {e}")
#         print(traceback.format_exc())
    
#     print(f"\nВсего методов загружено: {len(tester.methods)}")
    
#     # ============ ЗАГРУЗКА ТЕСТОВЫХ ДАННЫХ ============
    
#     print("\n" + "=" * 60)
#     print("ЗАГРУЗКА ТЕСТОВЫХ ДАННЫХ")
#     print("=" * 60)
    
#     # Загрузка тестового изображения
#     try:
#         repo_id = "hf-internal-testing/fixtures_ade20k"
#         print(f"Загрузка из репозитория: {repo_id}")
        
#         image_path = hf_hub_download(
#             repo_id=repo_id, 
#             filename="ADE_val_00000001.jpg", 
#             repo_type="dataset"
#         )
#         image = Image.open(image_path)
        
#         segmentation_map_path = hf_hub_download(
#             repo_id=repo_id, 
#             filename="ADE_val_00000001.png", 
#             repo_type="dataset"
#         )
#         segmentation_map = Image.open(segmentation_map_path)
        
#         print(f"✅ Изображение загружено: {image_path}")
#         print(f"   Размер: {image.size}")
#         print(f"✅ Ground truth загружен: {segmentation_map_path}")
        
#         # Сохраняем локально для тестов
#         local_image_path = "./data/test_image.jpg"
#         image.save(local_image_path)
#         print(f"✅ Изображение сохранено локально: {local_image_path}")

#         print("Для дополнительных тестов Используем тестовое изображение по умолчанию...")
#         img_url_1: str = "https://i.pinimg.com/736x/17/e7/fc/1D7oZ9cqSef531ErnBAai8ZivwSPyqMCcs.jpg"
#         local_image_path_1: str = "./data/test_image_download_1.jpg"
#         try:
#             response_1 = requests.get(img_url_1)
#             image_1 = Image.open(BytesIO(response_1.content))
#             image_1.save(local_image_path_1)
#             print(f"✅ Изображение загружено из URL: {img_url_1}")
#             print(f"   Размер: {image_1.size}")
#         except Exception as e1:
#             print(f"❌ Ошибка загрузки из URL: {e1}")
#             raise

#         img_url_2: str = "https://www.shutterstock.com/shutterstock/videos/1106252821/thumb/1.jpg?ip=x480"
#         local_image_path_2: str = "./data/test_image_download_2.jpg"
        
#         try:
#             response_2 = requests.get(img_url_2)
#             image_2 = Image.open(BytesIO(response_2.content))
#             image_2.save(local_image_path_2)
#             print(f"✅ Изображение загружено из URL: {img_url_2}")
#             print(f"   Размер: {image_2.size}")
#         except Exception as e2:
#             print(f"❌ Ошибка загрузки из URL: {e2}")
#             raise

#         img_url_3: str = "https://i.pinimg.com/736x/86/f6/07/86f60748d5d9ae4cb9092018d1321648.jpg"
#         local_image_path_3: str = "./data/test_image_download_3.jpg"
        
#         try:
#             response_3 = requests.get(img_url_3)
#             image_3 = Image.open(BytesIO(response_3.content))
#             image_3.save(local_image_path_3)
#             print(f"✅ Изображение загружено из URL: {img_url_3}")
#             print(f"   Размер: {image_3.size}")
#         except Exception as e3:
#             print(f"❌ Ошибка загрузки из URL: {e3}")
#             raise

#         img_url_4: str = "https://images.pond5.com/pov-car-and-truck-traffic-footage-190002081_iconl.jpeg"
#         local_image_path_4: str = "./data/test_image_download_4.jpg"
        
#         try:
#             response_4 = requests.get(img_url_4)
#             image_4 = Image.open(BytesIO(response_4.content))
#             image_4.save(local_image_path_4)
#             print(f"✅ Изображение загружено из URL: {img_url_4}")
#             print(f"   Размер: {image_4.size}")
#         except Exception as e4:
#             print(f"❌ Ошибка загрузки из URL: {e4}")
#             raise

#         img_url_5: str = "https://i.pinimg.com/736x/17/66/c4/1D7oZ9cqSef531ErnBAai8ZivwSPyqMCcs.jpg"
#         local_image_path_5: str = "./data/test_image_download_5.jpg"
        
#         try:
#             response_5 = requests.get(img_url_5)
#             image_5 = Image.open(BytesIO(response_5.content))
#             image_5.save(local_image_path_5)
#             print(f"✅ Изображение загружено из URL: {img_url_5}")
#             print(f"   Размер: {image_5.size}")
#         except Exception as e5:
#             print(f"❌ Ошибка загрузки из URL: {e5}")
#             raise

#         img_url_6: str = "https://i.pinimg.com/736x/f7/5a/f2/f75af26820b50c24600f50f3998eb02f.jpg"
#         local_image_path_6: str = "./data/test_image_download_6.jpg"
        
#         try:
#             response_6 = requests.get(img_url_6)
#             image_6 = Image.open(BytesIO(response_6.content))
#             image_6.save(local_image_path_6)
#             print(f"✅ Изображение загружено из URL: {img_url_6}")
#             print(f"   Размер: {image_6.size}")
#         except Exception as e6:
#             print(f"❌ Ошибка загрузки из URL: {e6}")
#             raise
        
#     except Exception as e:
#         print(f"❌ Ошибка загрузки из Hugging Face: {e}")
#         print("Используем тестовое изображение по умолчанию...")
        
#         # Альтернативный URL
#         img_url = "https://i.pinimg.com/736x/17/e7/fc/1D7oZ9cqSef531ErnBAai8ZivwSPyqMCcs.jpg"
#         local_image_path = "./data/test_image_default.jpg"
        
#         try:
#             response = requests.get(img_url)
#             image = Image.open(BytesIO(response.content))
#             image.save(local_image_path)
#             print(f"✅ Изображение загружено из URL: {img_url}")
#             print(f"   Размер: {image.size}")
#         except Exception as e2:
#             print(f"❌ Ошибка загрузки из URL: {e2}")
#             raise
    
#     # ============ ТЕСТИРОВАНИЕ НЕЙРОСЕТЕВОЙ СЕГМЕНТАЦИИ ============
    
#     print("\n" + "=" * 60)
#     print("ТЕСТИРОВАНИЕ НЕЙРОСЕТЕВОЙ СЕГМЕНТАЦИИ")
#     print("=" * 60)
    
#     if "Neural_SegFormer" in tester.methods:
#         try:
#             neural_segmenter = tester.methods["Neural_SegFormer"]
            
#             # Вариант 1: Простая сегментация (segment_image)
#             print("\n1. Простая сегментация (segment_image)...")
#             simple_result = neural_segmenter.segment_image(local_image_path, alpha=0.5)
            
#             # Сохраняем результат
#             simple_result.save("./data/neural_segmentation_result.jpg")
#             print(f"✅ Результат сохранен: neural_segmentation_result.jpg")

#             segmented_image_path_1 = "./data/test_segmented_image_1.jpg"
#             simple_result_1 = neural_segmenter.segment_image(local_image_path_1, alpha=0.5)
#             simple_result_1.save(segmented_image_path_1)
#             print(f"✅ Результат сохранен: {segmented_image_path_1}")

#             segmented_image_path_2 = "./data/test_segmented_image_2.jpg"
#             simple_result_2 = neural_segmenter.segment_image(local_image_path_2, alpha=0.5)
#             simple_result_2.save(segmented_image_path_2)
#             print(f"✅ Результат сохранен: {segmented_image_path_2}")

#             segmented_image_path_3 = "./data/test_segmented_image_3.jpg"
#             simple_result_3 = neural_segmenter.segment_image(local_image_path_3, alpha=0.5)
#             simple_result_3.save(segmented_image_path_3)
#             print(f"✅ Результат сохранен: {segmented_image_path_3}")

#             segmented_image_path_4 = "./data/test_segmented_image_4.jpg"
#             simple_result_4 = neural_segmenter.segment_image(local_image_path_4, alpha=0.5)
#             simple_result_4.save(segmented_image_path_4)
#             print(f"✅ Результат сохранен: {segmented_image_path_4}")

#             segmented_image_path_5 = "./data/test_segmented_image_5.jpg"
#             simple_result_5 = neural_segmenter.segment_image(local_image_path_5, alpha=0.5)
#             simple_result_5.save(segmented_image_path_5)
#             print(f"✅ Результат сохранен: {segmented_image_path_5}")

#             segmented_image_path_6 = "./data/test_segmented_image_6.jpg"
#             simple_result_6 = neural_segmenter.segment_image(local_image_path_6, alpha=0.5)
#             simple_result_6.save(segmented_image_path_6)
#             print(f"✅ Результат сохранен: {segmented_image_path_6}")
            
#             # ============ СРАВНЕНИЕ С GROUND TRUTH ============
        
#             print("\n2. Сравнение с Ground Truth...")
            
#             # Получаем палитру из нейросетевого сегментатора
#             palette = neural_segmenter.palette
#             palette_array = np.array(palette, dtype=np.uint8)
            
#             # 2.1 Ground Truth визуализация (как в оригинальном коде)
#             print("\n  2.1 Визуализация Ground Truth...")
#             ground_truth_seg = np.array(segmentation_map)  # 2D ground truth segmentation map
            
#             ground_truth_color_seg = np.zeros(
#                 (ground_truth_seg.shape[0], ground_truth_seg.shape[1], 3), 
#                 dtype=np.uint8
#             )
            
#             # Для ADE20K: ground truth начинается с 1, а предсказание с 0
#             # Поэтому используем ground_truth_seg - 1
#             for label, color in enumerate(palette_array):
#                 ground_truth_color_seg[ground_truth_seg - 1 == label, :] = color
            
#             # Конвертируем из BGR в RGB (если нужно)
#             ground_truth_color_seg = ground_truth_color_seg[..., ::-1]
            
#             # Создаем наложение ground truth на оригинал
#             ground_truth_overlay = np.array(image) * 0.5 + ground_truth_color_seg * 0.5
#             ground_truth_overlay = ground_truth_overlay.astype(np.uint8)

#             # Сохраняем ground truth overlay
#             ground_truth_overlay_img = Image.fromarray(ground_truth_overlay)
#             ground_truth_overlay_img.save("./data/ground_truth_overlay.jpg")
#             print(f"✅ Ground Truth Overlay сохранен: ground_truth_overlay.jpg")
            
#             # Сохраняем цветную сегментацию ground truth
#             ground_truth_color_img = Image.fromarray(ground_truth_color_seg)
#             ground_truth_color_img.save("./data/ground_truth_color.jpg")
#             print(f"✅ Ground Truth Color Map сохранен: ground_truth_color.jpg")
            
#             # 2.2 Neural Segmentation визуализация
#             print("\n  2.2 Визуализация Neural Segmentation...")
#             # Получаем сегментированное изображение с alpha=0.5 для сравнения
#             neural_result = neural_segmenter.segment_image(local_image_path, alpha=0.5)
#             neural_np = np.array(neural_result)
            
#             # Сохраняем neural segmentation с alpha=0.5
#             neural_result.save("./data/neural_segmentation_alpha_0.5.jpg")
#             print(f"✅ Neural Segmentation (alpha=0.5) сохранен: neural_segmentation_alpha_0.5.jpg")
            
#             # 2.3 Сравнительная визуализация
#             print("\n  2.3 Сравнительная визуализация...")
#             fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
#             # Оригинальное изображение
#             axes[0, 0].imshow(image)
#             axes[0, 0].set_title("Original Image")
#             axes[0, 0].axis('off')
            
#             # Neural Segmentation (alpha=1)
#             axes[0, 1].imshow(simple_result)
#             axes[0, 1].set_title("Neural Segmentation (alpha=1)")
#             axes[0, 1].axis('off')
            
#             # Neural Segmentation (alpha=0.5) - для сравнения с GT
#             axes[0, 2].imshow(neural_np)
#             axes[0, 2].set_title("Neural Segmentation (alpha=0.5)")
#             axes[0, 2].axis('off')
            
#             # Ground Truth (цветная сегментация)
#             axes[1, 0].imshow(ground_truth_color_seg)
#             axes[1, 0].set_title("Ground Truth Color Map")
#             axes[1, 0].axis('off')
            
#             # Ground Truth Overlay
#             axes[1, 1].imshow(ground_truth_overlay)
#             axes[1, 1].set_title("Ground Truth Overlay (alpha=0.5)")
#             axes[1, 1].axis('off')
            
#             # Side-by-side сравнение
#             comparison_img = np.hstack([neural_np, ground_truth_overlay])
#             axes[1, 2].imshow(comparison_img)
#             axes[1, 2].set_title("Comparison: Neural (left) vs GT (right)")
#             axes[1, 2].axis('off')
            
#             plt.suptitle("Neural Segmentation vs Ground Truth", fontsize=16)
#             plt.tight_layout()
            
#             # Сохраняем сводный график
#             plt.savefig("./data/neural_vs_gt_summary.jpg", dpi=150, bbox_inches='tight')
#             print(f"✅ Сводный график сохранен: neural_vs_gt_summary.jpg")
#             plt.show()
            
#             # ============ РАСЧЕТ МЕТРИК ============
            
#             print("\n3. Расчет метрик качества...")
            
#             # Получаем предсказанную карту сегментации
#             pred_seg_map = neural_segmenter.predict_segmentation_map(local_image_path)
#             pred_seg_map_1 = neural_segmenter.predict_segmentation_map(local_image_path_1)
#             pred_seg_map_2 = neural_segmenter.predict_segmentation_map(local_image_path_2)
#             pred_seg_map_3 = neural_segmenter.predict_segmentation_map(local_image_path_3)
#             pred_seg_map_4 = neural_segmenter.predict_segmentation_map(local_image_path_4)
#             pred_seg_map_5 = neural_segmenter.predict_segmentation_map(local_image_path_5)
#             pred_seg_map_6 = neural_segmenter.predict_segmentation_map(local_image_path_6)
            
#             # Сохраняем карту сегментации
#             predicted_segmentation_map_0 = "./data/prediction_segmentation_map.png"
#             pred_seg_map_normalized = (pred_seg_map / pred_seg_map.max() * 255).astype(np.uint8)
#             pred_seg_map_img = Image.fromarray(pred_seg_map_normalized)
#             pred_seg_map_img.save(predicted_segmentation_map_0)
#             print(f"✅ Карта сегментации сохранена: {predicted_segmentation_map_0}")

#             predicted_segmentation_map_1 = "./data/prediction_segmentation_map_1.png"
#             pred_seg_map_normalized_1 = (pred_seg_map_1 / pred_seg_map_1.max() * 255).astype(np.uint8)
#             pred_seg_map_img_1 = Image.fromarray(pred_seg_map_normalized_1)
#             pred_seg_map_img_1.save(predicted_segmentation_map_1)
#             print(f"✅ Карта сегментации сохранена: {predicted_segmentation_map_1}")

#             predicted_segmentation_map_2 = "./data/prediction_segmentation_map_2.png"
#             pred_seg_map_normalized_2 = (pred_seg_map_2 / pred_seg_map_2.max() * 255).astype(np.uint8)
#             pred_seg_map_img_2 = Image.fromarray(pred_seg_map_normalized_2)
#             pred_seg_map_img_2.save(predicted_segmentation_map_2)
#             print(f"✅ Карта сегментации сохранена: {predicted_segmentation_map_2}")

#             predicted_segmentation_map_3 = "./data/prediction_segmentation_map_3.png"
#             pred_seg_map_normalized_3 = (pred_seg_map_3 / pred_seg_map_3.max() * 255).astype(np.uint8)
#             pred_seg_map_img_3 = Image.fromarray(pred_seg_map_normalized_3)
#             pred_seg_map_img_3.save(predicted_segmentation_map_3)
#             print(f"✅ Карта сегментации сохранена: {predicted_segmentation_map_3}")

#             predicted_segmentation_map_4 = "./data/prediction_segmentation_map_4.png"
#             pred_seg_map_normalized_4 = (pred_seg_map_4 / pred_seg_map_4.max() * 255).astype(np.uint8)
#             pred_seg_map_img_4 = Image.fromarray(pred_seg_map_normalized_4)
#             pred_seg_map_img_4.save(predicted_segmentation_map_4)
#             print(f"✅ Карта сегментации сохранена: {predicted_segmentation_map_4}")

#             predicted_segmentation_map_5 = "./data/prediction_segmentation_map_5.png"
#             pred_seg_map_normalized_5 = (pred_seg_map_5 / pred_seg_map_5.max() * 255).astype(np.uint8)
#             pred_seg_map_img_5 = Image.fromarray(pred_seg_map_normalized_5)
#             pred_seg_map_img_5.save(predicted_segmentation_map_5)
#             print(f"✅ Карта сегментации сохранена: {predicted_segmentation_map_5}")

#             predicted_segmentation_map_6 = "./data/prediction_segmentation_map_6.png"
#             pred_seg_map_normalized_6 = (pred_seg_map_6 / pred_seg_map_6.max() * 255).astype(np.uint8)
#             pred_seg_map_img_6 = Image.fromarray(pred_seg_map_normalized_6)
#             pred_seg_map_img_6.save(predicted_segmentation_map_6)
#             print(f"✅ Карта сегментации сохранена: {predicted_segmentation_map_6}")
            
#             # Выравниваем размеры
#             h, w = min(pred_seg_map.shape[0], ground_truth_seg.shape[0]), \
#                 min(pred_seg_map.shape[1], ground_truth_seg.shape[1])
            
#             pred_flat = pred_seg_map[:h, :w].flatten()
#             gt_flat = (ground_truth_seg[:h, :w] - 1).flatten()  # -1 для соответствия
            
#             # Вычисляем метрики
#             try:
#                 from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                
#                 # Общие классы
#                 common_classes = np.intersect1d(np.unique(pred_flat), np.unique(gt_flat))
                
#                 if len(common_classes) > 0:
#                     accuracy = accuracy_score(gt_flat, pred_flat)
#                     precision = precision_score(gt_flat, pred_flat, average='weighted', zero_division=0)
#                     recall = recall_score(gt_flat, pred_flat, average='weighted', zero_division=0)
#                     f1 = f1_score(gt_flat, pred_flat, average='weighted', zero_division=0)
                    
#                     print(f"\n   Метрики качества:")
#                     print(f"   - Accuracy:  {accuracy:.4f}")
#                     print(f"   - Precision: {precision:.4f}")
#                     print(f"   - Recall:    {recall:.4f}")
#                     print(f"   - F1-Score:  {f1:.4f}")
#                     print(f"   - Общих классов: {len(common_classes)}")
                    
#                     # Сохраняем метрики в файл
#                     metrics_file = "./data/segmentation_metrics.txt"
#                     with open(metrics_file, 'w') as f:
#                         f.write("Segmentation Metrics\n")
#                         f.write("="*50 + "\n")
#                         f.write(f"Accuracy:  {accuracy:.4f}\n")
#                         f.write(f"Precision: {precision:.4f}\n")
#                         f.write(f"Recall:    {recall:.4f}\n")
#                         f.write(f"F1-Score:  {f1:.4f}\n")
#                         f.write(f"Common Classes: {len(common_classes)}\n")
#                         f.write(f"Image: {local_image_path}\n")
#                         f.write(f"Model: {neural_segmenter.model_name}\n")
#                         f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    
#                     print(f"✅ Метрики сохранены в: {metrics_file}")
                    
#                     # Визуализация разницы
#                     diff = np.abs(pred_seg_map[:h, :w] - (ground_truth_seg[:h, :w] - 1))
                    
#                     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    
#                     # Предсказание
#                     axes[0].imshow(pred_seg_map[:h, :w], cmap='tab20')
#                     axes[0].set_title(f"Prediction\nAccuracy: {accuracy:.3f}")
#                     axes[0].axis('off')
                    
#                     # Ground Truth
#                     axes[1].imshow(ground_truth_seg[:h, :w] - 1, cmap='tab20')
#                     axes[1].set_title("Ground Truth")
#                     axes[1].axis('off')
                    
#                     # Разность
#                     im = axes[2].imshow(diff, cmap='hot', vmin=0, vmax=10)
#                     axes[2].set_title("Difference (Prediction - GT)")
#                     axes[2].axis('off')
#                     plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
                    
#                     plt.suptitle(f"Segmentation Comparison - F1 Score: {f1:.3f}", fontsize=14)
#                     plt.tight_layout()
                    
#                     # Сохраняем график сравнения
#                     plt.savefig("./data/segmentation_comparison_detailed.jpg", dpi=150, bbox_inches='tight')
#                     print(f"✅ Детальный график сравнения сохранен: segmentation_comparison_detailed.jpg")
#                     plt.show()
                    
#                     # Сохраняем разность как изображение
#                     diff_normalized = (diff / diff.max() * 255).astype(np.uint8) if diff.max() > 0 else diff.astype(np.uint8)
#                     diff_img = Image.fromarray(diff_normalized)
#                     diff_img.save("./data/segmentation_difference.png")
#                     print(f"✅ Карта разности сохранена: segmentation_difference.png")
                    
#                 else:
#                     print("   ⚠️ Нет общих классов для сравнения")
                    
#             except ImportError:
#                 print("   ⚠️ scikit-learn не установлен. Пропускаем расчет метрик.")
#                 print("   Установите: pip install scikit-learn")

#             # ============ ДОПОЛНИТЕЛЬНЫЕ СОХРАНЕНИЯ ============
            
#             print("\n4. Дополнительные сохранения...")
#             import os
#             import shutil
            
#             # Сохраняем оригинальное изображение
#             image.save("./data/original_image.jpg")
#             print(f"✅ Оригинальное изображение сохранено: original_image.jpg")
            
#             # Сохраняем ground truth как есть
#             segmentation_map.save("./data/ground_truth_original.png")
#             print(f"✅ Ground Truth оригинал сохранен: ground_truth_original.png")
            
#             # Сохраняем все в отдельную папку
#             results_dir = "./data/neural_segmentation_results"
#             os.makedirs(results_dir, exist_ok=True)
            
#             # Перемещаем все файлы в папку
#             files_to_move = [
#                 "neural_segmentation_result.jpg",
#                 "neural_segmentation_alpha_0.5.jpg",
#                 "ground_truth_overlay.jpg",
#                 "ground_truth_color.jpg",
#                 "neural_vs_gt_summary.jpg",
#                 "segmentation_metrics.txt",
#                 "segmentation_comparison_detailed.jpg",
#                 "segmentation_difference.png",
#                 "prediction_segmentation_map.png",
#                 "original_image.jpg",
#                 "ground_truth_original.png"
#             ]
            
#             for file in files_to_move:
#                 if os.path.exists(file):
#                     shutil.move(file, os.path.join(results_dir, file))
#                     print(f"   Перемещен: {file} -> {results_dir}/{file}")
            
#             print(f"\n✅ Все результаты сохранены в папку: {results_dir}")
            
#             # ============ ФИНАЛЬНАЯ СВОДКА ============
            
#             print("\n" + "="*60)
#             print("ФИНАЛЬНАЯ СВОДКА РЕЗУЛЬТАТОВ")
#             print("="*60)
#             print(f"Папка с результатами: {results_dir}")
#             print(f"Исходное изображение: {local_image_path}")
#             print(f"Ground Truth: {segmentation_map_path}")
#             print(f"Модель: {neural_segmenter.model_name}")
            
#             if os.path.exists(os.path.join(results_dir, "segmentation_metrics.txt")):
#                 print("\nМетрики качества:")
#                 with open(os.path.join(results_dir, "segmentation_metrics.txt"), 'r') as f:
#                     print(f.read())

#             # ============ ДЕТАЛЬНАЯ СЕГМЕНТАЦИЯ ============

#             # Вариант 2: Детальная сегментация
#             print("\n2. Детальная сегментация...")
#             detailed_result = neural_segmenter.detailed_segmentation(local_image_path)
#             detailed_result_1 = neural_segmenter.detailed_segmentation(local_image_path_1)
#             detailed_result_2 = neural_segmenter.detailed_segmentation(local_image_path_2)
#             detailed_result_3 = neural_segmenter.detailed_segmentation(local_image_path_3)
#             detailed_result_4 = neural_segmenter.detailed_segmentation(local_image_path_4)
#             detailed_result_5 = neural_segmenter.detailed_segmentation(local_image_path_5)
#             detailed_result_6 = neural_segmenter.detailed_segmentation(local_image_path_6)

#             # Выводим информацию о классах
#             print(f"Обнаружено классов по изображению 0: {detailed_result['total_classes']}")
#             print(f"Обнаружено классов по изображению 1: {detailed_result_1['total_classes']}")
#             print(f"Обнаружено классов по изображению 2: {detailed_result_2['total_classes']}")
#             print(f"Обнаружено классов по изображению 3: {detailed_result_3['total_classes']}")
#             print(f"Обнаружено классов по изображению 4: {detailed_result_4['total_classes']}")
#             print(f"Обнаружено классов по изображению 5: {detailed_result_5['total_classes']}")
#             print(f"Обнаружено классов по изображению 6: {detailed_result_6['total_classes']}")

#             print("\nТоп-5 классов по площади (0):")

#             sorted_classes = sorted(detailed_result['class_distribution'].items(), 
#                                     key=lambda x: x[1]['pixel_count'], 
#                                     reverse=True)[:5]

#             for i, (class_name, info) in enumerate(sorted_classes, 1):
#                 print(f"  {i}. {class_name}: {info['percentage']:.1f}% ({info['pixel_count']} пикселей)")

#             # Сохраняем overlay изображение из detailed_result
#             neural_segmentation_result_detailed_0 = "./data/neural_segmentation_result_detailed.jpg"
#             # Извлекаем overlay из словаря и сохраняем
#             overlay_0 = Image.fromarray(detailed_result['overlay'])
#             overlay_0.save(neural_segmentation_result_detailed_0)
#             print(f"✅ Результат сохранен: {neural_segmentation_result_detailed_0}")

#             print("\nТоп-5 классов по площади (1):")
#             sorted_classes_1 = sorted(detailed_result_1['class_distribution'].items(), 
#                                     key=lambda x: x[1]['pixel_count'], 
#                                     reverse=True)[:5]

#             for i, (class_name, info) in enumerate(sorted_classes_1, 1):
#                 print(f"  {i}. {class_name}: {info['percentage']:.1f}% ({info['pixel_count']} пикселей)")

#             neural_segmentation_result_detailed_1 = "./data/neural_segmentation_result_detailed_1.jpg"
#             overlay_1 = Image.fromarray(detailed_result_1['overlay'])
#             overlay_1.save(neural_segmentation_result_detailed_1)
#             print(f"✅ Результат сохранен: {neural_segmentation_result_detailed_1}")

#             print("\nТоп-5 классов по площади (2):")
#             sorted_classes_2 = sorted(detailed_result_2['class_distribution'].items(), 
#                                     key=lambda x: x[1]['pixel_count'], 
#                                     reverse=True)[:5]

#             for i, (class_name, info) in enumerate(sorted_classes_2, 1):
#                 print(f"  {i}. {class_name}: {info['percentage']:.1f}% ({info['pixel_count']} пикселей)")

#             neural_segmentation_result_detailed_2 = "./data/neural_segmentation_result_detailed_2.jpg"
#             overlay_2 = Image.fromarray(detailed_result_2['overlay'])
#             overlay_2.save(neural_segmentation_result_detailed_2)
#             print(f"✅ Результат сохранен: {neural_segmentation_result_detailed_2}")

#             print("\nТоп-5 классов по площади (3):")
#             sorted_classes_3 = sorted(detailed_result_3['class_distribution'].items(), 
#                                     key=lambda x: x[1]['pixel_count'], 
#                                     reverse=True)[:5]

#             for i, (class_name, info) in enumerate(sorted_classes_3, 1):
#                 print(f"  {i}. {class_name}: {info['percentage']:.1f}% ({info['pixel_count']} пикселей)")

#             neural_segmentation_result_detailed_3 = "./data/neural_segmentation_result_detailed_3.jpg"
#             overlay_3 = Image.fromarray(detailed_result_3['overlay'])
#             overlay_3.save(neural_segmentation_result_detailed_3)
#             print(f"✅ Результат сохранен: {neural_segmentation_result_detailed_3}")

#             print("\nТоп-5 классов по площади (4):")
#             sorted_classes_4 = sorted(detailed_result_4['class_distribution'].items(), 
#                                     key=lambda x: x[1]['pixel_count'], 
#                                     reverse=True)[:5]

#             for i, (class_name, info) in enumerate(sorted_classes_4, 1):
#                 print(f"  {i}. {class_name}: {info['percentage']:.1f}% ({info['pixel_count']} пикселей)")

#             neural_segmentation_result_detailed_4 = "./data/neural_segmentation_result_detailed_4.jpg"
#             overlay_4 = Image.fromarray(detailed_result_4['overlay'])
#             overlay_4.save(neural_segmentation_result_detailed_4)
#             print(f"✅ Результат сохранен: {neural_segmentation_result_detailed_4}")

#             print("\nТоп-5 классов по площади (5):")
#             sorted_classes_5 = sorted(detailed_result_5['class_distribution'].items(), 
#                                     key=lambda x: x[1]['pixel_count'], 
#                                     reverse=True)[:5]

#             for i, (class_name, info) in enumerate(sorted_classes_5, 1):
#                 print(f"  {i}. {class_name}: {info['percentage']:.1f}% ({info['pixel_count']} пикселей)")

#             neural_segmentation_result_detailed_5 = "./data/neural_segmentation_result_detailed_5.jpg"
#             overlay_5 = Image.fromarray(detailed_result_5['overlay'])
#             overlay_5.save(neural_segmentation_result_detailed_5)
#             print(f"✅ Результат сохранен: {neural_segmentation_result_detailed_5}")

#             print("\nТоп-5 классов по площади (6):")
#             sorted_classes_6 = sorted(detailed_result_6['class_distribution'].items(), 
#                                     key=lambda x: x[1]['pixel_count'], 
#                                     reverse=True)[:5]

#             for i, (class_name, info) in enumerate(sorted_classes_6, 1):
#                 print(f"  {i}. {class_name}: {info['percentage']:.1f}% ({info['pixel_count']} пикселей)")

#             neural_segmentation_result_detailed_6 = "./data/neural_segmentation_result_detailed_6.jpg"
#             overlay_6 = Image.fromarray(detailed_result_6['overlay'])
#             overlay_6.save(neural_segmentation_result_detailed_6)
#             print(f"✅ Результат сохранен: {neural_segmentation_result_detailed_6}")

#             # Показываем overlay из detailed_segmentation_0
#             fig, axes = plt.subplots(1, 2, figsize=(12, 6))
#             axes[0].imshow(image)
#             axes[0].set_title("Original Image")
#             axes[0].axis('off')

#             axes[1].imshow(detailed_result['overlay'])
#             axes[1].set_title("Detailed Segmentation (alpha=0.5)")
#             axes[1].axis('off')

#             plt.tight_layout()
#             plt.show()
            
#         except Exception as e:
#             print(f"❌ Ошибка тестирования нейросетевой сегментации: {e}")
#             # print(traceback.format_exc())
                
#     else:
#         print("⚠️ Нейросетевая сегментация не доступна")
    
#     # ============ СРАВНЕНИЕ ВСЕХ МЕТОДОВ ============
    
#     print("\n" + "=" * 60)
#     print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ МЕТОДОВ СЕГМЕНТАЦИИ")
#     print("=" * 60)
    
#     # Выбираем подмножество методов для сравнения
#     selected_methods = [
#         "Global_Threshold_CV2",
#         "Adaptive_Threshold_CV2",
#         "Otsu_CV2",
#         "Region_Growing_CV2",
#         "Split_and_Merge_CV2",
#         "Sobel_CV2",
#         "Canny_CV2",
#         "KMeans_CV2",
#         # "DBSCAN_CV2",
#         "Active_Contour_CV2",
#         "GVF_Contour_CV2",
#         "Watershed_CV2",
#         # "Meanshift_CV2",
#         "GrabCut_CV2",
#         "FloodFill_CV2",
#         "Morphological_Snakes_CV2",
#         # "Quickshift_CV2",
#         "Slic_CV2",
#         "Felzenszwalb_CV2",
#         "Chan_Vese_CV2",
#         "Threshold_Niblack_CV2",
#         "Threshold_Sauvola_CV2",
#         "Random_Walker_CV2",
#         "Global_Threshold_Torch",
#         "Adaptive_Threshold_Torch",
#         "Otsu_Torch",
#         # "Region_Growing_Torch",
#         "Split_and_Merge_Torch",
#         "Sobel_Torch",
#         "Canny_Torch",
#         "KMeans_Torch",
#         # "DBSCAN_Torch",
#         # "Active_Contour_Torch",
#         # "GVF_Contour_Torch",
#         "Watershed_Torch",
#         # "Meanshift_Torch",
#         "Grabcut_Torch",
#         "FloodFill_Torch",
#     ]
    
#     if "Neural_SegFormer" in tester.methods:
#         selected_methods.append("Neural_SegFormer")
    
#     print(f"Сравниваем {len(selected_methods)} методов...")
    
#     try:
#         results = tester.compare_methods(
#             local_image_path, 
#             method_names=selected_methods,
#             figsize=(20, 15),
#             test_name="full_comparison_0",  # Имя теста
#             show_plots=True  # Показывать графики
#         )
        
#         # Визуализация сравнения
#         tester.visualize_comparison(results, 
#                                     show_masks=True,
#                                     save_visualization=True,
#                                     show_plots=True)

#         results_1 = tester.compare_methods(
#             local_image_path_1, 
#             method_names=selected_methods,
#             figsize=(20, 15),
#             test_name="full_comparison_1",
#             show_plots=True
#         )
#         tester.visualize_comparison(results_1, 
#                                     show_masks=True,
#                                     save_visualization=True,
#                                     show_plots=True)

#         results_2 = tester.compare_methods(
#             local_image_path_2, 
#             method_names=selected_methods,
#             figsize=(20, 15),
#             test_name="full_comparison_2",
#             show_plots=True
#         )
#         tester.visualize_comparison(results_2, 
#                                     show_masks=True,
#                                     save_visualization=True,
#                                     show_plots=True)

#         results_3 = tester.compare_methods(
#             local_image_path_3, 
#             method_names=selected_methods,
#             figsize=(20, 15),
#             test_name="full_comparison_3",
#             show_plots=True
#         )
#         tester.visualize_comparison(results_3, 
#                                     show_masks=True,
#                                     save_visualization=True,
#                                     show_plots=True)

#         results_4 = tester.compare_methods(
#             local_image_path_4, 
#             method_names=selected_methods,
#             figsize=(20, 15),
#             test_name="full_comparison_4",
#             show_plots=True
#         )
#         tester.visualize_comparison(results_4, 
#                                     show_masks=True,
#                                     save_visualization=True,
#                                     show_plots=True)

#         results_5 = tester.compare_methods(
#             local_image_path_5, 
#             method_names=selected_methods,
#             figsize=(20, 15),
#             test_name="full_comparison_5",
#             show_plots=True
#         )
#         tester.visualize_comparison(results_5, 
#                                     show_masks=True,
#                                     save_visualization=True,
#                                     show_plots=True)

#         results_6 = tester.compare_methods(
#             local_image_path_6,
#             method_names=selected_methods,
#             figsize=(20, 15),
#             test_name="full_comparison_6",
#             show_plots=True
#         )
#         tester.visualize_comparison(results_6, 
#                                     show_masks=True,
#                                     save_visualization=True,
#                                     show_plots=True)
        
#     except Exception as e:
#         print(f"❌ Ошибка при сравнении методов: {e}")
#         results = {}
#         results_1 = {}
#         results_2 = {}
#         results_3 = {}
#         results_4 = {}
#         results_5 = {}
#         results_6 = {}

#      # ============ Сравнение с таймингом ============
#     print("\n" + "=" * 60)
#     print("ТЕСТИРОВАНИЕ СРАВНЕНИЯ МЕТОДОВ С ТАЙМИНГОМ")
#     print("=" * 60)
    
#     try:
#         # Тестируем на последнем изображении
#         print(f"Тестируем на изображении (0): {local_image_path}")
        
#         # Запускаем сравнение с таймингом
#         methods_00, results_00, masks_00, times_00 = compare_segmentation_methods_with_timing(local_image_path)
#         dict_compare_00 = original_compare_segmentation_methods(local_image_path)

#         print(f"Тестируем на изображении (1): {local_image_path_1}")
#         methods_01, results_01, masks_01, times_01 = compare_segmentation_methods_with_timing(local_image_path_1)
#         dict_compare_01 = original_compare_segmentation_methods(local_image_path_1)

#         print(f"Тестируем на изображении (2): {local_image_path_2}")
#         methods_02, results_02, masks_02, times_02 = compare_segmentation_methods_with_timing(local_image_path_2)
#         dict_compare_02 = original_compare_segmentation_methods(local_image_path_2)

#         print(f"Тестируем на изображении (3): {local_image_path_3}")
#         methods_03, results_03, masks_03, times_03 = compare_segmentation_methods_with_timing(local_image_path_3)
#         dict_compare_03 = original_compare_segmentation_methods(local_image_path_3)

#         print(f"Тестируем на изображении (4): {local_image_path_4}")
#         methods_04, results_04, masks_04, times_04 = compare_segmentation_methods_with_timing(local_image_path_4)
#         dict_compare_04 = original_compare_segmentation_methods(local_image_path_4)

#         print(f"Тестируем на изображении (5): {local_image_path_5}")
#         methods_05, results_05, masks_05, times_05 = compare_segmentation_methods_with_timing(local_image_path_5)
#         dict_compare_05 = original_compare_segmentation_methods(local_image_path_5)

#         print(f"Тестируем на изображении (6): {local_image_path_6}")
#         methods_06, results_06, masks_06, times_06 = compare_segmentation_methods_with_timing(local_image_path_6)
#         dict_compare_06 = original_compare_segmentation_methods(local_image_path_6)
        
#         print("\n" + "=" * 60)
#         print("ТЕСТИРОВАНИЕ С ТАЙМИНГОМ ЗАВЕРШЕНО")
#         print("=" * 60)
        
#     except Exception as e:
#         print(f"❌ Ошибка при тестировании с таймингом: {e}")
#         import traceback
#         traceback.print_exc()
    
#     # ============ БЕНЧМАРК ПРОИЗВОДИТЕЛЬНОСТИ ============
    
#     print("\n" + "=" * 60)
#     print("БЕНЧМАРК ПРОИЗВОДИТЕЛЬНОСТИ")
#     print("=" * 60)
    
#     try:
#         df = tester.benchmark_methods(local_image_path, 
#                                       n_runs=2,
#                                       test_name="performance_test_0",
#                                       save_results=True)
#     except Exception as ex0:
#         print(f"❌ Ошибка бенчмарка (0): {ex0}")
#         df = pd.DataFrame()

#     try:
#         df_1 = tester.benchmark_methods(local_image_path_1, 
#                                         n_runs=2,
#                                         test_name="performance_test_1",
#                                         save_results=True)
#     except Exception as ex1:
#         print(f"❌ Ошибка бенчмарка (1): {ex1}")
#         df_1 = pd.DataFrame()


#     try:
#         df_2 = tester.benchmark_methods(local_image_path_2, 
#                                         n_runs=2,
#                                         test_name="performance_test_2",
#                                         save_results=True)
#     except Exception as ex2:
#         print(f"❌ Ошибка бенчмарка (2): {ex2}")
#         df_2 = pd.DataFrame() 

#     try:
#         df_3 = tester.benchmark_methods(local_image_path_3, 
#                                         n_runs=2,
#                                         test_name="performance_test_3",
#                                         save_results=True)
#     except Exception as ex3:
#         print(f"❌ Ошибка бенчмарка (3): {ex3}")
#         df_3 = pd.DataFrame()

#     try:
#         df_4 = tester.benchmark_methods(local_image_path_4, 
#                                         n_runs=2,
#                                         test_name="performance_test_4",
#                                         save_results=True)
#     except Exception as ex4:
#         print(f"❌ Ошибка бенчмарка (4): {ex4}")
#         df_4 = pd.DataFrame()
        
#     try:
#         df_5 = tester.benchmark_methods(local_image_path_5, 
#                                         n_runs=2,
#                                         test_name="performance_test_5",
#                                         save_results=True)
#     except Exception as ex5:
#         print(f"❌ Ошибка бенчмарка (5): {ex5}")
#         df_5 = pd.DataFrame()
    
#     try:
#         df_6 = tester.benchmark_methods(local_image_path_6, 
#                                         n_runs=2,
#                                         test_name="performance_test_6",
#                                         save_results=True)
#     except Exception as ex6:
#         print(f"❌ Ошибка бенчмарка (6): {ex6}")
#         df_6 = pd.DataFrame()    
    
    
#     # ============ СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ============
    
#     print("\n" + "=" * 60)
#     print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
#     print("=" * 60)
    
#     output_dir = "./data/segmentation_comparison_results"
#     try:
#         tester.save_results(results, output_dir)
        
#         # Дополнительно: сохраняем сводную таблицу
#         if not df.empty:
#             summary_path = os.path.join(output_dir, "summary.csv")
#             df.to_csv(summary_path, index=False)
#             print(f"✅ Сводная таблица сохранена (0): {summary_path}")
#     except Exception as ex0:
#         print(f"⚠️ Ошибка сохранения результатов: {ex0}")

#     try:
#         tester.save_results(results_1, output_dir)
#         if not df_1.empty:
#             summary_path = os.path.join(output_dir, "summary_1.csv")
#             df_1.to_csv(summary_path, index=False)
#             print(f"✅ Сводная таблица сохранена (1): {summary_path}")
#     except Exception as ex1:
#         print(f"⚠️ Ошибка сохранения результатов: {ex1}")

#     try:
#         tester.save_results(results_2, output_dir)
#         if not df_2.empty:
#             summary_path = os.path.join(output_dir, "summary_2.csv")
#             df_2.to_csv(summary_path, index=False)
#             print(f"✅ Сводная таблица сохранена (2): {summary_path}")
#     except Exception as ex2:
#         print(f"⚠️ Ошибка сохранения результатов: {ex2}")

#     try:
#         tester.save_results(results_3, output_dir)
#         if not df_3.empty:
#             summary_path = os.path.join(output_dir, "summary_3.csv")
#             df_3.to_csv(summary_path, index=False)
#             print(f"✅ Сводная таблица сохранена (3): {summary_path}")
#     except Exception as ex3:
#         print(f"⚠️ Ошибка сохранения результатов: {ex3}")

#     try:
#         tester.save_results(results_4, output_dir)
#         if not df_4.empty:
#             summary_path = os.path.join(output_dir, "summary_4.csv")
#             df_4.to_csv(summary_path, index=False)
#             print(f"✅ Сводная таблица сохранена (4): {summary_path}")
#     except Exception as ex4:
#         print(f"⚠️ Ошибка сохранения результатов: {ex4}")

#     try:
#         tester.save_results(results_5, output_dir)
#         if not df_5.empty:
#             summary_path = os.path.join(output_dir, "summary_5.csv")
#             df_5.to_csv(summary_path, index=False)
#             print(f"✅ Сводная таблица сохранена (5): {summary_path}")
#     except Exception as ex5:
#         print(f"⚠️ Ошибка сохранения результатов: {ex5}")

#     try:
#         tester.save_results(results_6, output_dir)
#         if not df_6.empty:
#             summary_path = os.path.join(output_dir, "summary_6.csv")
#             df_6.to_csv(summary_path, index=False)
#             print(f"✅ Сводная таблица сохранена (6): {summary_path}")
#     except Exception as ex6:
#         print(f"⚠️ Ошибка сохранения результатов: {ex6}")

#     # ============ ВЫВОД ИНФОРМАЦИИ ============
    
#     print("\n" + "=" * 60)
#     print("РЕЗЮМЕ")
#     print("=" * 60)
#     print(f"✓ Загружено методов: {len(tester.methods)}")
#     print(f"✓ Протестировано: {len(selected_methods)}")
#     print(f"✓ Изображение (0): {local_image_path}")
#     print(f"✓ Изображение (1): {local_image_path_1}")
#     print(f"✓ Изображение (2): {local_image_path_2}")
#     print(f"✓ Изображение (3): {local_image_path_3}")
#     print(f"✓ Изображение (4): {local_image_path_4}")
#     print(f"✓ Изображение (5): {local_image_path_5}")
#     print(f"✓ Изображение (6): {local_image_path_6}")
#     print(f"✓ Результаты сохранены в: {output_dir}")
#     print("=" * 60)

#     pipeline = create_advanced_pipeline_example()
    
#     print(f"Создан пайплайн с шагами: {pipeline.get_step_names()}")
    
#     # Запускаем пайплайн на тестовом изображении
#     # (нужно иметь test_image.jpg в директории)
#     try:
#         results = pipeline.run("./data/test_image_download_6.jpg", visualize=True)
        
#         # Анализируем результаты
#         analysis = analyze_pipeline_results(results)
#         print_pipeline_analysis(analysis)
        
#         # Сохраняем визуализацию
#         pipeline.save_visualization(results, "./data/test_image_download_6.jpg", "./data/advanced_pipeline_results.jpg")
        
#     except FileNotFoundError:
#         print("⚠️ Файл test_image_download_6.jpg не найден.")
#         print("Доступные файлы:")
#         import os
#         files = [f for f in os.listdir('.') if f.endswith('.jpg')]
#         for f in files:
#             print(f"  - {f}")
#     except Exception as e:
#         print(f"❌ Ошибка выполнения пайплайна: {e}")
#         import traceback
#         traceback.print_exc()

#     # ============ ЗАГРУЗКА ИЗОБРАЖЕНИЙ ============

#     print("=" * 60)
#     print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ МЕТОДОВ СЕГМЕНТАЦИИ")
#     print("=" * 60)
    
#     # Загрузка нескольких тестовых изображений
#     test_images = load_test_images()
    
#     if not test_images:
#         print("❌ Нет тестовых изображений!")
#         return tester, {}, pd.DataFrame()
    
#     # ============ СЛУЧАЙ 1: СРАВНЕНИЕ БЕЗ GROUND TRUTH ============
    
#     print("\n" + "=" * 60)
#     print("СЛУЧАЙ 1: СРАВНЕНИЕ БЕЗ GROUND TRUTH")
#     print("=" * 60)
    
#     for img_name, (image_path, image, gt_mask) in test_images.items():
#         print(f"\n📷 Обработка изображения: {img_name}")
        
#         # Проверяем наличие ground truth
#         has_ground_truth = gt_mask is not None
        
#         if has_ground_truth:
#             print("  ✅ Ground truth доступен")
#             results_0 = test_with_ground_truth(tester, image_path, gt_mask, img_name)
#         else:
#             print("  ⚠️ Ground truth не доступен")
#             results_0 = test_without_ground_truth(tester, image_path, image, img_name)

#     create_summary_report(results_0, "final_summary_report")

#     print("\n" + "=" * 60)
#     print("✅ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
#     print("=" * 60)
#     print("\nРезультаты сохранены в:")
#     print("  - results_with_gt_*/ (тесты с ground truth)")
#     print("  - matrix_comparison_*/ (матричные сравнения)")
#     print("  - reference_comparison_*/ (сравнение с референсами)")
#     print("  - consistency_analysis_*/ (анализ согласованности)")
#     print("  - stability_analysis_*/ (анализ стабильности)")
#     print("  - final_summary_report/ (сводный отчет)")
#     print("\nДля просмотра отчетов откройте соответствующие HTML файлы.")

#     return tester, results, df

# def load_test_images() -> Dict[str, Tuple[str, Image.Image, Optional[np.ndarray]]]:
#     """
#     Загружает тестовые изображения. Возвращает словарь:
#     {
#         'имя_изображения': (путь, PIL.Image, ground_truth_mask или None)
#     }
#     """
#     import requests
#     from io import BytesIO
    
#     test_images = {}
    
#     # Примеры изображений с возможными ground truth
#     image_urls = {
#         "countryside": "https://i.pinimg.com/736x/17/e7/fc/1D7oZ9cqSef531ErnBAai8ZivwSPyqMCcs.jpg",
#         "nature": "https://i.pinimg.com/736x/f7/5a/f2/f75af26820b50c24600f50f3998eb02f.jpg",
#         "architecture": "https://i.pinimg.com/736x/86/f6/07/86f60748d5d9ae4cb9092018d1321648.jpg",
#         "trucks": "https://www.shutterstock.com/shutterstock/videos/1106252821/thumb/1.jpg?ip=x480",
#         "traffic": "https://images.pond5.com/pov-car-and-truck-traffic-footage-190002081_iconl.jpeg",
#         "mountain": "https://i.pinimg.com/736x/17/66/c4/1D7oZ9cqSef531ErnBAai8ZivwSPyqMCcs.jpg"
#     }
    
#     for name, url in image_urls.items():
#         try:
#             response = requests.get(url, timeout=10)
#             img = Image.open(BytesIO(response.content)).convert('RGB')
            
#             # Сохраняем локально
#             local_path = f"test_image_{name}.jpg"
#             img.save(local_path)
            
#             # Для примера, будем считать что ground truth нет
#             # На практике здесь можно было бы загрузить ground truth если он есть
#             test_images[name] = (local_path, img, None)
            
#             print(f"✅ {name}: {img.size}, ground truth: {'да' if None else 'нет'}")
            
#         except Exception as e:
#             print(f"❌ Ошибка загрузки {name}: {e}")
    
#     return test_images

# def test_with_ground_truth(tester: SegmentationTester, 
#                           image_path: str,
#                           ground_truth: np.ndarray,
#                           test_name: str) -> Dict[str, Any]:
#     """
#     Тестирование с доступным ground truth
#     """
#     print(f"\n🔬 Тестирование с Ground Truth: {test_name}")
    
#     # Инициализируем тестер с ground truth
#     tester_with_gt = SegmentationTester(
#         base_output_dir=f"results_with_gt_{test_name}",
#         ground_truth_path=None  # Передаем прямо маску
#     )
#     tester_with_gt.ground_truth_mask = ground_truth
    
#     # Добавляем методы для сравнения
#     add_segmentation_methods(tester_with_gt)
    
#     # Выполняем сравнение с метриками
#     results = tester_with_gt.compare_methods_with_metrics(
#         image=image_path,
#         method_names=get_methods_for_comparison(),
#         test_name=f"gt_comparison_{test_name}",
#         show_plots=True
#     )
    
#     return results

# def test_without_ground_truth(tester: SegmentationTester,
#                              image_path: str,
#                              image: Image.Image,
#                              test_name: str) -> Dict[str, Any]:
#     """
#     Тестирование без ground truth
#     Используем несколько стратегий сравнения
#     """
#     print(f"\n🎯 Тестирование без Ground Truth: {test_name}")
    
#     # Стратегия 1: Сравнение всех методов между собой
#     print("\n1. Матричное сравнение методов между собой...")
#     results_matrix = compare_methods_matrix(image_path, test_name)
    
#     # Стратегия 2: Использование референсных методов (sklearn/scikit-image)
#     print("\n2. Сравнение с референсными реализациями...")
#     results_reference = compare_with_reference_implementations(image, test_name)
    
#     # Стратегия 3: Оценка внутренней согласованности
#     print("\n3. Оценка внутренней согласованности...")
#     results_consistency = evaluate_consistency(image_path, test_name)
    
#     # Стратегия 4: Сравнение на нескольких изображениях
#     print("\n4. Кросс-валидация на нескольких изображениях...")
#     results_cross = cross_image_comparison(image_path, test_name)
    
#     return {
#         'matrix_comparison': results_matrix,
#         'reference_comparison': results_reference,
#         'consistency_analysis': results_consistency,
#         'cross_validation': results_cross
#     }

# def compare_methods_matrix(image_path: str, test_name: str) -> Dict[str, Any]:
#     """
#     Матричное сравнение всех методов между собой без ground truth
#     """
#     # Используем ExtendedSegmentationComparator
#     comparator = ExtendedSegmentationComparator()
    
#     # Конфигурация методов для сравнения
#     methods_config = [
#         {"name": "kmeans", "type": "sklearn", "params": {"n_clusters": 3}},
#         {"name": "dbscan", "type": "sklearn", "params": {"eps": 0.5, "min_samples": 5}},
#         {"name": "meanshift", "type": "sklearn", "params": {"bandwidth": 0.5}},
#         {"name": "felzenszwalb", "type": "skimage", "params": {"scale": 100, "sigma": 0.8}},
#         {"name": "slic", "type": "skimage", "params": {"n_segments": 100}},
#         {"name": "watershed", "type": "skimage", "params": {}},
#         {"name": "threshold_otsu", "type": "skimage", "params": {}},
#         {"name": "canny", "type": "skimage", "params": {"sigma": 1.0}},
#     ]
    
#     # Загружаем изображение
#     image = cv2.imread(image_path)
#     if image is None:
#         print(f"❌ Не удалось загрузить изображение: {image_path}")
#         return {}
    
#     # Выполняем матричное сравнение
#     results = comparator.matrix_comparison(
#         image=image,
#         methods_config=methods_config,
#         comparison_type="all_vs_all",
#         output_dir=f"matrix_comparison_{test_name}",
#         save_results=True
#     )
    
#     # Анализируем результаты
#     if 'df_comparisons' in results:
#         df = results['df_comparisons']
        
#         print(f"\n📊 Матричное сравнение завершено:")
#         print(f"  - Сравнено пар: {len(df)}")
#         print(f"  - Методов: {len(results.get('masks', {}))}")
        
#         # Находим наиболее похожие методы
#         if not df.empty and 'f1_score' in df.columns:
#             # Исключаем сравнение с самим собой
#             valid_comparisons = df[df['method1'] != df['method2']]
            
#             if not valid_comparisons.empty:
#                 most_similar = valid_comparisons.nlargest(3, 'f1_score')
#                 print(f"\n  Наиболее похожие методы (высокий F1):")
#                 for _, row in most_similar.iterrows():
#                     print(f"    {row['method1']} vs {row['method2']}: F1={row['f1_score']:.3f}")
                
#                 least_similar = valid_comparisons.nsmallest(3, 'f1_score')
#                 print(f"\n  Наиболее разные методы (низкий F1):")
#                 for _, row in least_similar.iterrows():
#                     print(f"    {row['method1']} vs {row['method2']}: F1={row['f1_score']:.3f}")
    
#     return results

# def compare_with_reference_implementations(image: Image.Image, test_name: str) -> Dict[str, Any]:
#     """
#     Сравнение кастомных реализаций с референсными (sklearn/scikit-image)
#     """
    
#     # Создаем компаратор
#     comparator = SegmentationComparator()
    
#     # Конвертируем PIL в numpy
#     img_np = np.array(image)
    
#     # Сравниваем несколько методов
#     comparisons = []
    
#     # Список сравнений: (кастомный_метод, референсный_метод, параметры)
#     comparison_pairs = [
#         ("kmeans_segmentation", "kmeans", {"k": 3}),
#         ("otsu_thresholding", "threshold_otsu", {}),
#         ("watershed", "watershed", {}),
#         ("canny_edge", "canny", {"low": 50, "high": 150}),
#     ]
    
#     for custom_method, ref_method, params in comparison_pairs:
#         try:
#             # Кастомная реализация
#             custom_segmenter = CV2SklearnSegmenter(custom_method, **params)
#             custom_mask, _ = custom_segmenter.segment_with_mask(img_np)
            
#             # Референсная реализация
#             if ref_method in ["kmeans", "dbscan", "meanshift", "gmm"]:
#                 ref_mask, ref_info = comparator.segment_with_sklearn(
#                     img_np, ref_method, **params)
#             else:
#                 ref_mask, ref_info = comparator.segment_with_skimage(
#                     img_np, ref_method, **params)
            
#             # Вычисляем метрики
#             metrics = comparator.compute_metrics(
#                 custom_mask, ref_mask, 
#                 f"Custom_{custom_method}", 
#                 f"Reference_{ref_method}"
#             )
            
#             comparisons.append({
#                 'custom_method': custom_method,
#                 'reference_method': ref_method,
#                 'metrics': metrics,
#                 'custom_mask': custom_mask,
#                 'reference_mask': ref_mask
#             })
            
#             print(f"  ✅ {custom_method} vs {ref_method}: "
#                   f"F1={metrics.get('f1_score', 0):.3f}, "
#                   f"IoU={metrics.get('jaccard', 0):.3f}")
            
#         except Exception as e:
#             print(f"  ❌ Ошибка сравнения {custom_method} vs {ref_method}: {e}")
    
#     # Визуализация результатов
#     if comparisons:
#         visualize_reference_comparisons(comparisons, img_np, test_name)
    
#     return {'comparisons': comparisons}

# def visualize_reference_comparisons(comparisons: List[Dict], 
#                                   original_image: np.ndarray,
#                                   test_name: str):
#     """Визуализирует сравнение кастомных и референсных методов"""
#     n_comparisons = len(comparisons)
    
#     if n_comparisons == 0:
#         return
    
#     fig, axes = plt.subplots(n_comparisons, 4, figsize=(16, n_comparisons * 4))
    
#     if n_comparisons == 1:
#         axes = axes.reshape(1, -1)
    
#     for i, comparison in enumerate(comparisons):
#         custom_mask = comparison['custom_mask']
#         ref_mask = comparison['reference_mask']
#         metrics = comparison['metrics']
        
#         # Оригинал
#         if len(original_image.shape) == 2:
#             axes[i, 0].imshow(original_image, cmap='gray')
#         else:
#             axes[i, 0].imshow(original_image)
#         axes[i, 0].set_title(f"{comparison['custom_method']}\nOriginal")
#         axes[i, 0].axis('off')
        
#         # Кастомная маска
#         axes[i, 1].imshow(custom_mask, cmap='gray')
#         axes[i, 1].set_title(f"Custom\n{comparison['custom_method']}")
#         axes[i, 1].axis('off')
        
#         # Референсная маска
#         axes[i, 2].imshow(ref_mask, cmap='gray')
#         axes[i, 2].set_title(f"Reference\n{comparison['reference_method']}")
#         axes[i, 2].axis('off')
        
#         # Разность
#         diff = np.abs(custom_mask.astype(float) - ref_mask.astype(float))
#         im = axes[i, 3].imshow(diff, cmap='hot')
#         axes[i, 3].set_title(f"Difference\nF1={metrics.get('f1_score', 0):.3f}")
#         axes[i, 3].axis('off')
        
#         # Цветовая шкала для разности
#         plt.colorbar(im, ax=axes[i, 3], fraction=0.046, pad=0.04)
    
#     plt.suptitle(f"Сравнение кастомных и референсных реализаций - {test_name}", 
#                  fontsize=14)
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
#     output_dir = f"reference_comparison_{test_name}"
#     os.makedirs(output_dir, exist_ok=True)
    
#     plt.savefig(os.path.join(output_dir, "comparison_summary.jpg"), 
#                 dpi=150, bbox_inches='tight')
#     plt.close()
    
#     print(f"📊 Визуализация сохранена в {output_dir}/")

# def evaluate_consistency(image_path: str, test_name: str) -> Dict[str, Any]:
#     """
#     Оценивает внутреннюю согласованность методов без ground truth
#     """
    
#     # Загружаем изображение
#     image = cv2.imread(image_path)
#     if image is None:
#         return {}
    
#     # Методы для оценки
#     methods = {
#         "Global_Threshold_CV2": CV2SklearnSegmenter("global_thresholding", threshold=127),
#         "Adaptive_Threshold_CV2": CV2SklearnSegmenter("adaptive_thresholding", block_size=11, C=2),
#         "Otsu_CV2": CV2SklearnSegmenter("otsu_thresholding"),
#         "Region_Growing_CV2": CV2SklearnSegmenter("region_growing", seed=(100, 100), tolerance=20),
#         "Split_and_Merge_CV2": CV2SklearnSegmenter("split_and_merge", min_region_size=50, threshold=20),
#         "Sobel_CV2": CV2SklearnSegmenter("sobel_edge", threshold=50),
#         "Canny_CV2": CV2SklearnSegmenter("canny_edge", low=50, high=150),
#         "KMeans_CV2": CV2SklearnSegmenter("kmeans_segmentation", k=3),
#         # "DBSCAN_CV2": CV2SklearnSegmenter("dbscan_segmentation", eps=10, min_samples=100),
#         "Active_Contour_CV2": CV2SklearnSegmenter("active_contour", alpha=0.01, beta=0.1, gamma=0.001, max_iterations=2000),
#         "GVF_Contour_CV2": CV2SklearnSegmenter("gvf_contour", mu=0.2, iterations=100),
#         "Watershed_CV2": CV2SklearnSegmenter("watershed"),
#         # "Meanshift_CV2": CV2SklearnSegmenter("meanshift", bandwidth=0.5),
#         "GrabCut_CV2": CV2SklearnSegmenter("grabcut", iterations=10),
#         "FloodFill_CV2": CV2SklearnSegmenter("floodfill", seed=(100, 100), tolerance=20),
#         "Morphological_Snakes_CV2": CV2SklearnSegmenter("morphological_snakes", iterations=100, smoothing=1, threshold=0.5),
#         # "Quickshift_CV2": CV2SklearnSegmenter("quickshift", bandwidth=0.5),
#         "Slic_CV2": CV2SklearnSegmenter("slic", n_segments=100, compactness=10.0),
#         "Felzenszwalb_CV2": CV2SklearnSegmenter("felzenszwalb", scale=100, sigma=0.8, min_size=50),
#         "Chan_Vese_CV2": CV2SklearnSegmenter("chan_vese", mu=0.25, lambda1=1.0, lambda2=1.0, tol=1e-3, max_iter=100),
#         "Threshold_Niblack_CV2": CV2SklearnSegmenter("threshold_niblack", window_size=15, k=0.2),
#         "Threshold_Sauvola_CV2": CV2SklearnSegmenter("threshold_sauvola", window_size=15, k=0.2, r=128),
#         "Random_Walker_CV2": CV2SklearnSegmenter("random_walker", scale=100, sigma=0.8, min_size=50),
#     }
    
#     # Получаем маски от всех методов
#     masks = {}
#     execution_times = {}
    
#     for name, segmenter in methods.items():
#         try:
#             start_time = datetime.now()
#             mask, _ = segmenter.segment_with_mask(image)
#             exec_time = (datetime.now() - start_time).total_seconds()
            
#             masks[name] = mask
#             execution_times[name] = exec_time
            
#             print(f"  ✅ {name}: {exec_time:.3f}s")
#         except Exception as e:
#             print(f"  ❌ {name}: {e}")
    
#     # Вычисляем метрики согласованности
#     consistency_metrics = {}
    
#     if len(masks) > 1:
#         method_names = list(masks.keys())
#         n_methods = len(method_names)
        
#         # Матрица попарного согласия
#         agreement_matrix = np.zeros((n_methods, n_methods))
        
#         for i, m1 in enumerate(method_names):
#             for j, m2 in enumerate(method_names):
#                 if i == j:
#                     agreement_matrix[i, j] = 1.0
#                 else:
#                     # Простое согласие по пикселям
#                     mask1_bin = (masks[m1] > 127).astype(np.uint8).flatten()
#                     mask2_bin = (masks[m2] > 127).astype(np.uint8).flatten()
                    
#                     agreement = np.mean(mask1_bin == mask2_bin)
#                     agreement_matrix[i, j] = agreement
        
#         # Вычисляем среднее согласие для каждого метода
#         mean_agreement = {}
#         for i, m in enumerate(method_names):
#             # Исключаем само согласие
#             other_indices = [j for j in range(n_methods) if j != i]
#             mean_agreement[m] = np.mean(agreement_matrix[i, other_indices])
        
#         # Сводная статистика
#         consistency_metrics = {
#             'agreement_matrix': agreement_matrix,
#             'mean_agreement': mean_agreement,
#             'method_names': method_names,
#             'execution_times': execution_times,
#             'overall_mean_agreement': np.mean(list(mean_agreement.values())),
#             'overall_std_agreement': np.std(list(mean_agreement.values()))
#         }
        
#         # Визуализация матрицы согласия
#         visualize_consistency_matrix(agreement_matrix, method_names, test_name)
        
#         print(f"\n📊 Метрики согласованности:")
#         print(f"  Среднее согласие: {consistency_metrics['overall_mean_agreement']:.3f}")
#         print(f"  Стандартное отклонение: {consistency_metrics['overall_std_agreement']:.3f}")
        
#         print(f"\n  Согласие по методам:")
#         for method, agreement in sorted(mean_agreement.items(), 
#                                        key=lambda x: x[1], reverse=True):
#             print(f"    {method}: {agreement:.3f}")
    
#     return consistency_metrics

# def visualize_consistency_matrix(matrix: np.ndarray, 
#                                method_names: List[str],
#                                test_name: str):
#     """Визуализирует матрицу согласия методов"""
#     fig, ax = plt.subplots(figsize=(10, 8))
    
#     # Сокращаем имена для подписей
#     short_names = [name[:10] for name in method_names]
    
#     im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1)
#     ax.set_xticks(range(len(method_names)))
#     ax.set_yticks(range(len(method_names)))
#     ax.set_xticklabels(short_names, rotation=45, ha='right')
#     ax.set_yticklabels(short_names)
    
#     # Добавляем значения в ячейки
#     for i in range(len(method_names)):
#         for j in range(len(method_names)):
#             text = ax.text(j, i, f"{matrix[i, j]:.2f}",
#                          ha="center", va="center",
#                          color="black" if matrix[i, j] < 0.7 else "white",
#                          fontsize=9)
    
#     ax.set_title("Матрица согласия методов", fontsize=14)
#     plt.colorbar(im, ax=ax, label='Согласие')
#     plt.tight_layout()
    
#     output_dir = f"consistency_analysis_{test_name}"
#     os.makedirs(output_dir, exist_ok=True)
    
#     plt.savefig(os.path.join(output_dir, "agreement_matrix.jpg"),
#                 dpi=150, bbox_inches='tight')
#     plt.close()
    
#     print(f"📈 Матрица согласия сохранена в {output_dir}/")

# def cross_image_comparison(image_path: str, test_name: str) -> Dict[str, Any]:
#     """
#     Кросс-валидация на нескольких изображениях
#     """
#     import glob
    
#     # Ищем другие изображения в директории
#     image_dir = os.path.dirname(image_path) or "."
#     all_images = glob.glob(os.path.join(image_dir, "*.jpg")) + \
#                  glob.glob(os.path.join(image_dir, "*.png")) + \
#                  glob.glob(os.path.join(image_dir, "*.jpeg"))
    
#     # Ограничиваем количество изображений для скорости
#     max_images = 5
#     test_images = [image_path] + all_images[:max_images-1] if len(all_images) > 1 else [image_path]
    
#     print(f"\n📸 Кросс-валидация на {len(test_images)} изображениях")
    
#     # Методы для тестирования
#     methods = {
#         "Global_Threshold_CV2": lambda: CV2SklearnSegmenter("global_thresholding", threshold=127),
#         "Adaptive_Threshold_CV2": lambda: CV2SklearnSegmenter("adaptive_thresholding", block_size=11, C=2),
#         "Otsu_CV2": lambda: CV2SklearnSegmenter("otsu_thresholding"),
#         "Region_Growing_CV2": lambda: CV2SklearnSegmenter("region_growing", seed=(100, 100), tolerance=20),
#         "Split_and_Merge_CV2": lambda: CV2SklearnSegmenter("split_and_merge", min_region_size=50, threshold=20),
#         "Sobel_CV2": lambda: CV2SklearnSegmenter("sobel_edge", threshold=50),
#         "Canny_CV2": lambda: CV2SklearnSegmenter("canny_edge", low=50, high=150),
#         "KMeans_CV2": lambda: CV2SklearnSegmenter("kmeans_segmentation", k=3),
#         # "DBSCAN_CV2":lambda: CV2SklearnSegmenter("dbscan_segmentation", eps=10, min_samples=100),
#         "Active_Contour_CV2": lambda: CV2SklearnSegmenter("active_contour", alpha=0.01, beta=0.1, gamma=0.001, max_iterations=2000),
#         "GVF_Contour_CV2": lambda: CV2SklearnSegmenter("gvf_contour", mu=0.2, iterations=100),
#         "Watershed_CV2": lambda: CV2SklearnSegmenter("watershed"),
#         # "Meanshift_CV2": lambda: CV2SklearnSegmenter("meanshift", bandwidth=0.5),
#         "GrabCut_CV2": lambda: CV2SklearnSegmenter("grabcut", iterations=10),
#         "FloodFill_CV2": lambda: CV2SklearnSegmenter("floodfill", seed=(100, 100), tolerance=20),
#         "Morphological_Snakes_CV2": lambda: CV2SklearnSegmenter("morphological_snakes", iterations=100, smoothing=1, threshold=0.5),
#         # "Quickshift_CV2": lambda: CV2SklearnSegmenter("quickshift", bandwidth=0.5),
#         "Slic_CV2": lambda: CV2SklearnSegmenter("slic", n_segments=100, compactness=10.0),
#         "Felzenszwalb_CV2": lambda: CV2SklearnSegmenter("felzenszwalb", scale=100, sigma=0.8, min_size=50),
#         "Chan_Vese_CV2": lambda: CV2SklearnSegmenter("chan_vese", mu=0.25, lambda1=1.0, lambda2=1.0, tol=1e-3, max_iter=100),
#         "Threshold_Niblack_CV2": lambda: CV2SklearnSegmenter("threshold_niblack", window_size=15, k=0.2),
#         "Threshold_Sauvola_CV2": lambda: CV2SklearnSegmenter("threshold_sauvola", window_size=15, k=0.2, r=128),
#         "Random_Walker_CV2": lambda: CV2SklearnSegmenter("random_walker", scale=100, sigma=0.8, min_size=50),
#     }
    
#     # Собираем результаты по всем изображениям
#     all_results = []
    
#     for img_idx, img_path in enumerate(test_images):
#         try:
#             image = cv2.imread(img_path)
#             if image is None:
#                 continue
            
#             img_name = os.path.basename(img_path)
#             print(f"  Обработка {img_name}...")
            
#             img_results = {'image': img_name, 'methods': {}}
            
#             for method_name, method_factory in methods.items():
#                 try:
#                     segmenter = method_factory()
#                     mask, _ = segmenter.segment_with_mask(image)
                    
#                     # Базовые метрики маски
#                     mask_binary = mask > 127
#                     area = np.sum(mask_binary)
#                     coverage = area / mask_binary.size * 100
                    
#                     img_results['methods'][method_name] = {
#                         'area': int(area),
#                         'coverage': float(coverage),
#                         'mask_shape': mask.shape
#                     }
                    
#                 except Exception as e:
#                     print(f"    ❌ {method_name} на {img_name}: {e}")
            
#             all_results.append(img_results)
            
#         except Exception as e:
#             print(f"  ❌ Ошибка обработки {img_path}: {e}")
    
#     # Анализируем стабильность методов
#     stability_analysis = {}
    
#     for method_name in methods.keys():
#         # Собираем покрытия для данного метода на всех изображениях
#         coverages = []
        
#         for img_result in all_results:
#             if method_name in img_result['methods']:
#                 coverages.append(img_result['methods'][method_name]['coverage'])
        
#         if coverages:
#             stability_analysis[method_name] = {
#                 'mean_coverage': np.mean(coverages),
#                 'std_coverage': np.std(coverages),
#                 'cv_coverage': np.std(coverages) / np.mean(coverages) * 100 if np.mean(coverages) > 0 else 0,
#                 'n_images': len(coverages)
#             }
    
#     # Визуализируем стабильность
#     visualize_stability_analysis(stability_analysis, test_name)
    
#     print(f"\n📊 Анализ стабильности методов:")
#     for method_name, stats in stability_analysis.items():
#         print(f"  {method_name}:")
#         print(f"    Среднее покрытие: {stats['mean_coverage']:.1f}%")
#         print(f"    Стандартное отклонение: {stats['std_coverage']:.1f}%")
#         print(f"    Коэффициент вариации: {stats['cv_coverage']:.1f}%")
    
#     return {
#         'all_results': all_results,
#         'stability_analysis': stability_analysis,
#         'n_images_processed': len(all_results)
#     }

# def visualize_stability_analysis(stability_analysis: Dict[str, Any], test_name: str):
#     """Визуализирует анализ стабильности методов"""
#     if not stability_analysis:
#         return
    
#     methods = list(stability_analysis.keys())
#     mean_coverages = [stats['mean_coverage'] for stats in stability_analysis.values()]
#     std_coverages = [stats['std_coverage'] for stats in stability_analysis.values()]
    
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
#     # График 1: Среднее покрытие с ошибками
#     x_pos = np.arange(len(methods))
#     ax1.bar(x_pos, mean_coverages, yerr=std_coverages, 
#            capsize=5, alpha=0.7, color='skyblue')
#     ax1.set_xticks(x_pos)
#     ax1.set_xticklabels(methods, rotation=45)
#     ax1.set_ylabel('Покрытие (%)')
#     ax1.set_title('Среднее покрытие маски ± стандартное отклонение')
#     ax1.grid(True, alpha=0.3)
    
#     # Добавляем значения на столбцы
#     for i, (mean, std) in enumerate(zip(mean_coverages, std_coverages)):
#         ax1.text(i, mean + std + 1, f'{mean:.1f}±{std:.1f}', 
#                 ha='center', va='bottom', fontsize=9)
    
#     # График 2: Коэффициент вариации
#     cv_values = [stats['cv_coverage'] for stats in stability_analysis.values()]
#     ax2.bar(x_pos, cv_values, alpha=0.7, color='lightcoral')
#     ax2.set_xticks(x_pos)
#     ax2.set_xticklabels(methods, rotation=45)
#     ax2.set_ylabel('Коэффициент вариации (%)')
#     ax2.set_title('Коэффициент вариации покрытия (ниже = стабильнее)')
#     ax2.grid(True, alpha=0.3)
    
#     # Добавляем значения на столбцы
#     for i, cv in enumerate(cv_values):
#         ax2.text(i, cv + 0.5, f'{cv:.1f}%', 
#                 ha='center', va='bottom', fontsize=9)
    
#     plt.suptitle(f'Анализ стабильности методов сегментации - {test_name}', fontsize=14)
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
#     output_dir = f"stability_analysis_{test_name}"
#     os.makedirs(output_dir, exist_ok=True)
    
#     plt.savefig(os.path.join(output_dir, "stability_analysis.jpg"),
#                 dpi=150, bbox_inches='tight')
#     plt.close()
    
#     print(f"📈 Анализ стабильности сохранен в {output_dir}/")

# def add_segmentation_methods(tester: SegmentationTester):
#     """Добавляет методы сегментации в тестер"""
    
#     print("\nДобавление методов сегментации...")
    
#     # CV2/Sklearn методы
#     cv2_methods = {
#         "Global_Threshold_CV2": CV2SklearnSegmenter("global_thresholding", threshold=127),
#         "Adaptive_Threshold_CV2": CV2SklearnSegmenter("adaptive_thresholding", block_size=11, C=2),
#         "Otsu_CV2": CV2SklearnSegmenter("otsu_thresholding"),
#         "Region_Growing_CV2": CV2SklearnSegmenter("region_growing", seed=(100, 100), tolerance=20),
#         "Split_and_Merge_CV2": CV2SklearnSegmenter("split_and_merge", min_region_size=50, threshold=20),
#         "Sobel_CV2": CV2SklearnSegmenter("sobel_edge", threshold=50),
#         "Canny_CV2": CV2SklearnSegmenter("canny_edge", low=50, high=150),
#         "KMeans_CV2": CV2SklearnSegmenter("kmeans_segmentation", k=3),
#         # "DBSCAN_CV2": CV2SklearnSegmenter("dbscan_segmentation", eps=10, min_samples=100),
#         "Active_Contour_CV2": CV2SklearnSegmenter("active_contour", alpha=0.01, beta=0.1, gamma=0.001, max_iterations=2000),
#         "GVF_Contour_CV2": CV2SklearnSegmenter("gvf_contour", mu=0.2, iterations=100),
#         "Watershed_CV2": CV2SklearnSegmenter("watershed"),
#         # "Meanshift_CV2": CV2SklearnSegmenter("meanshift", bandwidth=0.5),
#         "GrabCut_CV2": CV2SklearnSegmenter("grabcut", iterations=10),
#         "FloodFill_CV2": CV2SklearnSegmenter("floodfill", seed=(100, 100), tolerance=20),
#         "Morphological_Snakes_CV2": CV2SklearnSegmenter("morphological_snakes", iterations=100, smoothing=1, threshold=0.5),
#         # "Quickshift_CV2": CV2SklearnSegmenter("quickshift", bandwidth=0.5),
#         "Slic_CV2": CV2SklearnSegmenter("slic", n_segments=100, compactness=10.0),
#         "Felzenszwalb_CV2": CV2SklearnSegmenter("felzenszwalb", scale=100, sigma=0.8, min_size=50),
#         "Chan_Vese_CV2": CV2SklearnSegmenter("chan_vese", mu=0.25, lambda1=1.0, lambda2=1.0, tol=1e-3, max_iter=100),
#         "Threshold_Niblack_CV2": CV2SklearnSegmenter("threshold_niblack", window_size=15, k=0.2),
#         "Threshold_Sauvola_CV2": CV2SklearnSegmenter("threshold_sauvola", window_size=15, k=0.2, r=128),
#         "Random_Walker_CV2": CV2SklearnSegmenter("random_walker", scale=100, sigma=0.8, min_size=50),
#     }
    
#     for name, segmenter in cv2_methods.items():
#         tester.add_method(name, segmenter)
#         print(f"  ✅ {name}")
    
#     # PyTorch методы (если доступны)
#     try:
#         torch_methods = {
#             "Global_Threshold_Torch": TorchSegmenter("global_thresholding", threshold=0.5),
#             "Adaptive_Threshold_Torch": TorchSegmenter("adaptive_thresholding", block_size=11, C=2),
#             "Otsu_Torch": TorchSegmenter("otsu_thresholding"),
#             "Sobel_Torch": TorchSegmenter("sobel_edge", threshold=0.1),
#             "Canny_Torch": TorchSegmenter("canny_edge", low=0.1, high=0.3),
#             "KMeans_Torch": TorchSegmenter("kmeans_segmentation", k=3),
#             "Watershed_Torch": TorchSegmenter("watershed"),
#         }
        
#         for name, segmenter in torch_methods.items():
#             tester.add_method(name, segmenter)
#             print(f"  ✅ {name}")
#     except Exception as e:
#         print(f"  ⚠️ PyTorch методы недоступны: {e}")
    
#     # Нейросетевая сегментация (если доступна)
#     try:
#         neural_segmenter = NeuralSegmenter(
#             local_path="/home/yamshchikov/models/segformer-b5-ready"
#         )
#         tester.add_method("Neural_SegFormer", neural_segmenter)
#         print(f"  ✅ Neural_SegFormer")
#     except Exception as e:
#         print(f"  ⚠️ Нейросетевая сегментация недоступна: {e}")

# def get_methods_for_comparison() -> List[str]:
#     """Возвращает список методов для сравнения"""
#     return [
#         "Global_Threshold_CV2",
#         "Otsu_CV2",
#         "KMeans_CV2",
#         "Watershed_CV2",
#         "Canny_CV2",
#         "GrabCut_CV2",
#     ]

# def create_summary_report(test_results: Dict[str, Any], output_dir: str = "summary_report"):
#     """
#     Создает сводный отчет по всем тестам
#     """
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Создаем HTML отчет
#     html_path = os.path.join(output_dir, "summary_report.html")
    
#     with open(html_path, 'w', encoding='utf-8') as f:
#         f.write("""
#         <!DOCTYPE html>
#         <html>
#         <head>
#             <title>Сводный отчет по сегментации</title>
#             <style>
#                 body { font-family: Arial, sans-serif; margin: 20px; }
#                 h1, h2, h3 { color: #333; }
#                 .summary { background: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
#                 .test-case { margin-bottom: 30px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
#                 .metrics-table { width: 100%; border-collapse: collapse; margin: 10px 0; }
#                 .metrics-table th, .metrics-table td { border: 1px solid #ddd; padding: 8px; text-align: center; }
#                 .metrics-table th { background-color: #f2f2f2; }
#                 .good { background-color: #d4edda; }
#                 .medium { background-color: #fff3cd; }
#                 .poor { background-color: #f8d7da; }
#                 .image-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; }
#                 .image-grid img { max-width: 100%; height: auto; border: 1px solid #ddd; }
#             </style>
#         </head>
#         <body>
#             <h1>📊 Сводный отчет по тестированию методов сегментации</h1>
#             <div class="summary">
#                 <p><strong>Дата тестирования:</strong> """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
#                 <p><strong>Всего тестов:</strong> """ + str(len(test_results)) + """</p>
#             </div>
#         """)
        
#         # Добавляем результаты по каждому тесту
#         for test_name, test_data in test_results.items():
#             f.write(f"""
#             <div class="test-case">
#                 <h2>{test_name}</h2>
#             """)
            
#             # Добавляем информацию о тесте
#             if 'has_ground_truth' in test_data:
#                 if test_data['has_ground_truth']:
#                     f.write("<p><strong>Тип:</strong> С Ground Truth</p>")
#                 else:
#                     f.write("<p><strong>Тип:</strong> Без Ground Truth</p>")
            
#             f.write("</div>")
        
#         f.write("""
#             <footer>
#                 <p>Сгенерировано автоматически. Для детальной информации смотрите папки с результатами.</p>
#             </footer>
#         </body>
#         </html>
#         """)
    
#     print(f"📄 Сводный отчет создан: {html_path}")

# # Дополнительные вспомогательные функции

# def evaluate_clustering_quality(image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
#     """
#     Оценивает качество кластеризации для сегментационных методов
#     """
#     # Преобразуем в признаки для кластеризации
#     if len(image.shape) == 3:
#         # Используем цветовые особенности
#         features = image.reshape(-1, 3)
#     else:
#         # Используем интенсивность и градиенты
#         from skimage.filters import sobel
#         gradient = sobel(image)
#         features = np.column_stack([image.flatten(), gradient.flatten()])
    
#     # Метки из маски
#     labels = (mask > 127).flatten().astype(int)
    
#     metrics = {}
    
#     try:
#         # Оцениваем только если есть как минимум 2 кластера
#         if len(np.unique(labels)) >= 2:
#             metrics['silhouette_score'] = silhouette_score(features, labels)
#             metrics['calinski_harabasz_score'] = calinski_harabasz_score(features, labels)
#             metrics['davies_bouldin_score'] = davies_bouldin_score(features, labels)
#     except:
#         # Если не удалось вычислить метрики
#         pass
    
#     return metrics

# def analyze_method_robustness(image_paths: List[str], method_factory):
#     """
#     Анализирует робастность метода на нескольких изображениях
#     """
#     results = []
    
#     for img_path in image_paths:
#         try:
#             image = cv2.imread(img_path)
#             if image is None:
#                 continue
            
#             segmenter = method_factory()
#             mask, _ = segmenter.segment_with_mask(image)
            
#             # Базовые метрики
#             mask_binary = mask > 127
#             area = np.sum(mask_binary)
#             coverage = area / mask_binary.size * 100
            
#             # Качество кластеризации
#             clustering_metrics = evaluate_clustering_quality(image, mask)
            
#             results.append({
#                 'image': os.path.basename(img_path),
#                 'area': area,
#                 'coverage': coverage,
#                 'clustering_metrics': clustering_metrics
#             })
            
#         except Exception as e:
#             print(f"Ошибка обработки {img_path}: {e}")
    
#     return results

# def test_neural_segmentation_variants():
#     """Тестирование различных вариантов нейросетевой сегментации"""
    
#     print("\n" + "=" * 60)
#     print("ТЕСТИРОВАНИЕ ВАРИАНТОВ НЕЙРОСЕТЕВОЙ СЕГМЕНТАЦИИ")
#     print("=" * 60)
    
#     # Загрузка тестового изображения
#     img_url = "https://i.pinimg.com/736x/17/e7/fc/1D7oZ9cqSef531ErnBAai8ZivwSPyqMCcs.jpg"
    
#     try:
#         response = requests.get(img_url)
#         test_image = Image.open(BytesIO(response.content))
        
#         # Создаем нейросетевой сегментатор
#         segmenter = NeuralSegmenter(
#             local_path="/home/yamshchikov/models/segformer-b5-ready"
#         )
        
#         # Вариант 1: Различные значения alpha
#         print("\n1. Тестирование разных значений alpha:")
#         alphas = [0.3, 0.5, 0.7, 1.0]
        
#         fig, axes = plt.subplots(2, 2, figsize=(12, 10))
#         axes = axes.flatten()
        
#         for i, alpha in enumerate(alphas):
#             result = segmenter.segment_image(test_image, alpha=alpha)
#             axes[i].imshow(result)
#             axes[i].set_title(f"alpha = {alpha}")
#             axes[i].axis('off')
            
#             # Сохраняем
#             result.save(f"./data/neural_alpha_{alpha}.jpg")
        
#         plt.suptitle("Neural Segmentation with Different Alpha Values", fontsize=14)
#         plt.tight_layout()
#         plt.show()
        
#         # Вариант 2: Детальный анализ
#         print("\n2. Детальный анализ сегментации:")
#         detailed_result = segmenter.detailed_segmentation(test_image)
        
#         # Выводим информацию о классах
#         print(f"Обнаружено классов: {detailed_result['total_classes']}")
#         print("\nТоп-5 классов по площади:")
        
#         sorted_classes = sorted(detailed_result['class_distribution'].items(), 
#                                key=lambda x: x[1]['pixel_count'], 
#                                reverse=True)[:5]
        
#         for class_name, info in sorted_classes:
#             print(f"  {class_name}: {info['percentage']:.1f}% ({info['pixel_count']} пикселей)")
        
#         # Вариант 3: segment_with_mask
#         print("\n3. Тестирование segment_with_mask:")
#         result_np, mask = segmenter.segment_with_mask(test_image, alpha=0.5)
        
#         fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
#         axes[0].imshow(test_image)
#         axes[0].set_title("Original")
#         axes[0].axis('off')
        
#         axes[1].imshow(result_np)
#         axes[1].set_title("Segmentation Result")
#         axes[1].axis('off')
        
#         axes[2].imshow(mask, cmap='gray')
#         axes[2].set_title("Binary Mask")
#         axes[2].axis('off')
        
#         plt.tight_layout()
#         plt.show()
        
#         print(f"Размер маски: {mask.shape}")
#         print(f"Площадь маски: {np.sum(mask > 0)} пикселей")
        
#         return segmenter, detailed_result
        
#     except Exception as e:
#         print(f"❌ Ошибка тестирования нейросетевых вариантов: {e}")
#         print(traceback.format_exc())
#         return None, None

# def test_with_metrics():
#     """Тестирование с метриками качества"""
    
#     print("\n" + "="*60)
#     print("ТЕСТИРОВАНИЕ С МЕТРИКАМИ КАЧЕСТВА")
#     print("="*60)

#     repo_id = "hf-internal-testing/fixtures_ade20k"
#     print(f"Загрузка из репозитория: {repo_id}")

#     image_path = hf_hub_download(
#         repo_id=repo_id, 
#         filename="ADE_val_00000001.jpg", 
#         repo_type="dataset"
#     )
#     image = Image.open(image_path)
    
#     segmentation_map_path = hf_hub_download(
#         repo_id=repo_id, 
#         filename="ADE_val_00000001.png", 
#         repo_type="dataset"
#     )
#     segmentation_map = Image.open(segmentation_map_path)
    
#     print(f"✅ Изображение загружено: {image_path}")
#     print(f"   Размер: {image.size}")
#     print(f"✅ Ground truth загружен: {segmentation_map_path}")
    
#     # Сохраняем локально для тестов
#     local_image_path = "test_image_0.jpg"
#     image.save(local_image_path)
#     print(f"✅ Изображение сохранено локально: {local_image_path}")

#     local_mask_path = "test_mask_image_0.png"
#     segmentation_map.save(local_mask_path)
#     print(f"✅ Изображение маски сохранено локально: {local_mask_path}")
    
#     # Инициализация тестера с ground truth
#     tester = SegmentationTester(
#         base_output_dir="segmentation_metrics_results",
#         ground_truth_path=local_mask_path  # Путь к вашей ground truth маске
#     )
    
#     # Добавление методов
#     cv2_methods = {
#         "Global_Threshold_CV2": CV2SklearnSegmenter("global_thresholding", threshold=127),
#         "Adaptive_Threshold_CV2": CV2SklearnSegmenter("adaptive_thresholding", block_size=11, C=2),
#         "Otsu_CV2": CV2SklearnSegmenter("otsu_thresholding"),
#         "Region_Growing_CV2": CV2SklearnSegmenter("region_growing", seed=(100, 100), tolerance=20),
#         "Split_and_Merge_CV2": CV2SklearnSegmenter("split_and_merge", min_region_size=50, threshold=20),
#         "Sobel_CV2": CV2SklearnSegmenter("sobel_edge", threshold=50),
#         "Canny_CV2": CV2SklearnSegmenter("canny_edge", low=50, high=150),
#         "KMeans_CV2": CV2SklearnSegmenter("kmeans_segmentation", k=3),
#         # "DBSCAN_CV2": CV2SklearnSegmenter("dbscan_segmentation", eps=10, min_samples=100),
#         "Active_Contour_CV2": CV2SklearnSegmenter("active_contour", alpha=0.01, beta=0.1, gamma=0.001, max_iterations=2000),
#         "GVF_Contour_CV2": CV2SklearnSegmenter("gvf_contour", mu=0.2, iterations=100),
#         "Watershed_CV2": CV2SklearnSegmenter("watershed"),
#         # "Meanshift_CV2": CV2SklearnSegmenter("meanshift", bandwidth=0.5),
#         "GrabCut_CV2": CV2SklearnSegmenter("grabcut", iterations=10),
#         "FloodFill_CV2": CV2SklearnSegmenter("floodfill", seed=(100, 100), tolerance=20),
#         "Morphological_Snakes_CV2": CV2SklearnSegmenter("morphological_snakes", iterations=100, smoothing=1, threshold=0.5),
#         # "Quickshift_CV2": CV2SklearnSegmenter("quickshift", bandwidth=0.5),
#         "Slic_CV2": CV2SklearnSegmenter("slic", n_segments=100, compactness=10.0),
#         "Felzenszwalb_CV2": CV2SklearnSegmenter("felzenszwalb", scale=100, sigma=0.8, min_size=50),
#         "Chan_Vese_CV2": CV2SklearnSegmenter("chan_vese", mu=0.25, lambda1=1.0, lambda2=1.0, tol=1e-3, max_iter=100),
#         "Threshold_Niblack_CV2": CV2SklearnSegmenter("threshold_niblack", window_size=15, k=0.2),
#         "Threshold_Sauvola_CV2": CV2SklearnSegmenter("threshold_sauvola", window_size=15, k=0.2, r=128),
#         "Random_Walker_CV2": CV2SklearnSegmenter("random_walker", scale=100, sigma=0.8, min_size=50),
#     }
    
#     for name, segmenter in cv2_methods.items():
#         tester.add_method(name, segmenter)
#         print(f"   ✅ {name}")


#     neural_segmenter = NeuralSegmenter(
#         local_path="/home/yamshchikov/models/segformer-b5-ready"
#     )
#     tester.add_method("Neural_SegFormer", neural_segmenter)
#     print(f"   ✅ Neural_SegFormer Added")

#     method_names = ["Global_Threshold_CV2",
#                     "Adaptive_Threshold_CV2",
#                     "Otsu_CV2",
#                     "Region_Growing_CV2",
#                     "Split_and_Merge_CV2",
#                     "Sobel_CV2", 
#                     "Canny_CV2",
#                     "KMeans_CV2",
#                     "Active_Contour_CV2",
#                     "GVF_Contour_CV2",
#                     "GVF_Contour_CV2",
#                     "Watershed_CV2",
#                     "GrabCut_CV2",
#                     "FloodFill_CV2",
#                     "Morphological_Snakes_CV2",
#                     "Slic_CV2",
#                     "Felzenszwalb_CV2",
#                     "Chan_Vese_CV2",
#                     "Threshold_Niblack_CV2",
#                     "Threshold_Sauvola_CV2",
#                     "Random_Walker_CV2",
#                     "Neural_SegFormer"
#                     ]
    
#     # Сравнение с метриками
#     results = tester.compare_methods_with_metrics(
#         image=local_image_path,
#         method_names=method_names,
#         test_name="metrics_comparison",
#         show_plots=True
#     )
    
#     return tester, results

# def batch_evaluation(image_dir: str, gt_dir: str):
#     """
#     Пакетная оценка методов на наборе изображений
    
#     Args:
#         image_dir: Директория с изображениями
#         gt_dir: Директория с ground truth масками
#     """
#     import glob
    
#     # Получаем список изображений
#     image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
#     gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.png")))
    
#     if len(image_files) != len(gt_files):
#         print(f"⚠️ Количество изображений ({len(image_files)}) и масок ({len(gt_files)}) не совпадает")
#         return
    
#     # Создаем сегментаторы
#     segmenters = {
#         "Otsu": CV2SklearnSegmenter("otsu_thresholding"),
#         "Watershed": CV2SklearnSegmenter("watershed"),
#         "KMeans": CV2SklearnSegmenter("kmeans_segmentation", k=3),
#         "Neural": NeuralSegmenter(
#             local_path="/home/yamshchikov/models/segformer-b5-ready"
#         )
#     }
    
#     all_results = {}
    
#     # Оцениваем каждый метод
#     for method_name, segmenter in segmenters.items():
#         print(f"\nОценка метода: {method_name}")
        
#         pred_masks = []
#         gt_masks = []
        
#         for img_file, gt_file in zip(image_files, gt_files):
#             try:
#                 # Сегментируем
#                 pred_mask = segmenter.segment(img_file)
                
#                 # Загружаем ground truth
#                 gt_mask = cv2.imread(gt_file, cv2.IMREAD_GRAYSCALE)
                
#                 # Конвертируем pred_mask в uint8
#                 if pred_mask.dtype != np.uint8:
#                     if pred_mask.max() <= 1.0:
#                         pred_mask = (pred_mask * 255).astype(np.uint8)
#                     else:
#                         pred_mask = pred_mask.astype(np.uint8)
                
#                 pred_masks.append(pred_mask)
#                 gt_masks.append(gt_mask)
                
#             except Exception as e:
#                 print(f"  Ошибка обработки {os.path.basename(img_file)}: {e}")
        
#         # Оцениваем метрики на всех изображениях
#         if pred_masks and gt_masks:
#             evaluation = SegmentationMetrics.evaluate_multiple_masks(
#                 pred_masks, gt_masks
#             )
#             all_results[method_name] = evaluation
            
#             print(f"  Средний IoU: {evaluation['average_metrics']['avg_iou']:.3f}")
#             print(f"  Средний Dice: {evaluation['average_metrics']['avg_dice']:.3f}")
    
#     # Сохраняем результаты
#     output_dir = "batch_evaluation_results"
#     os.makedirs(output_dir, exist_ok=True)
    
#     import json
#     with open(os.path.join(output_dir, "batch_evaluation.json"), 'w') as f:
#         json.dump(all_results, f, indent=2)
    
#     print(f"\n✅ Результаты сохранены в {output_dir}")
#     return all_results

# if __name__ == "__main__":
#     # Основной тест
#     print("ЗАПУСК ОСНОВНОГО ТЕСТА")
#     print("=" * 60)
#     tester, results, df = main()
#     tester, results = test_with_metrics()
#     # Дополнительный тест нейросетевых вариантов
#     # print("\n\nЗАПУСК ДОПОЛНИТЕЛЬНОГО ТЕСТА НЕЙРОСЕТЕВЫХ ВАРИАНТОВ")
#     # print("=" * 60)
#     # segmenter, detailed_result = test_neural_segmentation_variants()


# def compare_with_ground_truth_simple(image_path: str, 
#                                      gt_path: str, 
#                                      neural_segmenter: NeuralSegmenter,
#                                      save_results: bool = True):
#     """
#     Простое сравнение нейросетевой сегментации с ground truth
#     """
#     print("\n" + "="*60)
#     print("ПРОСТОЕ СРАВНЕНИЕ С GROUND TRUTH")
#     print("="*60)
    
#     try:
#         # Загрузка изображений
#         image = Image.open(image_path)
#         segmentation_map = Image.open(gt_path)
        
#         print(f"Изображение: {image_path}")
#         print(f"Ground Truth: {gt_path}")
#         print(f"Размер изображения: {image.size}")
#         print(f"Размер GT: {segmentation_map.size}")
        
#         # 1. Получаем палитру
#         palette = neural_segmenter.palette
#         palette_array = np.array(palette, dtype=np.uint8)
        
#         # 2. Ground Truth визуализация
#         ground_truth_seg = np.array(segmentation_map)
        
#         # Проверяем диапазон значений
#         print(f"\nДиапазон значений Ground Truth: {ground_truth_seg.min()} - {ground_truth_seg.max()}")
        
#         ground_truth_color_seg = np.zeros(
#             (ground_truth_seg.shape[0], ground_truth_seg.shape[1], 3), 
#             dtype=np.uint8
#         )
        
#         # Для ADE20K: ground truth начинается с 1
#         for label, color in enumerate(palette_array):
#             ground_truth_color_seg[ground_truth_seg - 1 == label, :] = color
        
#         # Конвертируем в RGB
#         ground_truth_color_seg = ground_truth_color_seg[..., ::-1]
        
#         # 3. Neural Segmentation
#         neural_result = neural_segmenter.segment_image(image_path, alpha=0.5)
#         neural_np = np.array(neural_result)
        
#         # 4. Создаем ground truth overlay
#         ground_truth_overlay = np.array(image) * 0.5 + ground_truth_color_seg * 0.5
#         ground_truth_overlay = ground_truth_overlay.astype(np.uint8)
        
#         # 5. Визуализация
#         fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
#         # Оригинал
#         axes[0, 0].imshow(image)
#         axes[0, 0].set_title("Original Image")
#         axes[0, 0].axis('off')
        
#         # Neural Segmentation
#         axes[0, 1].imshow(neural_np)
#         axes[0, 1].set_title("Neural Segmentation (alpha=0.5)")
#         axes[0, 1].axis('off')
        
#         # Ground Truth
#         axes[1, 0].imshow(ground_truth_overlay)
#         axes[1, 0].set_title("Ground Truth Overlay (alpha=0.5)")
#         axes[1, 0].axis('off')
        
#         # Разность
#         pred_seg_map = neural_segmenter.predict_segmentation_map(image_path)
        
#         # Выравниваем размеры
#         h, w = min(pred_seg_map.shape[0], ground_truth_seg.shape[0]), \
#                min(pred_seg_map.shape[1], ground_truth_seg.shape[1])
        
#         # Вычисляем разность
#         pred_resized = pred_seg_map[:h, :w]
#         gt_resized = ground_truth_seg[:h, :w] - 1
        
#         diff = np.abs(pred_resized - gt_resized)
        
#         # Нормализуем для визуализации
#         if diff.max() > 0:
#             diff_normalized = diff / diff.max()
#         else:
#             diff_normalized = diff
        
#         im = axes[1, 1].imshow(diff_normalized, cmap='hot')
#         axes[1, 1].set_title("Difference (Prediction vs GT)")
#         axes[1, 1].axis('off')
#         plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
        
#         plt.suptitle("Neural Segmentation vs Ground Truth", fontsize=14)
#         plt.tight_layout()
        
#         if save_results:
#             plt.savefig("./data/neural_vs_gt_comparison.jpg", dpi=150, bbox_inches='tight')
#             print(f"✅ Сравнение сохранено: neural_vs_gt_comparison.jpg")
        
#         plt.show()
        
#         # 6. Простые метрики
#         print("\nСтатистика:")
#         print(f"  - Размер предсказания: {pred_seg_map.shape}")
#         print(f"  - Размер Ground Truth: {ground_truth_seg.shape}")
#         print(f"  - Классы в предсказании: {len(np.unique(pred_seg_map))}")
#         print(f"  - Классы в GT: {len(np.unique(ground_truth_seg))}")
        
#         # Простой подсчет совпадений
#         if h > 0 and w > 0:
#             matches = np.sum(pred_resized == gt_resized)
#             total_pixels = h * w
#             match_percentage = (matches / total_pixels) * 100
            
#             print(f"  - Совпадение пикселей: {matches}/{total_pixels} ({match_percentage:.1f}%)")
        
#         return {
#             'image': image,
#             'ground_truth': ground_truth_seg,
#             'prediction': neural_result,
#             'prediction_map': pred_seg_map,
#             'gt_overlay': ground_truth_overlay
#         }
        
#     except Exception as e:
#         print(f"❌ Ошибка сравнения с Ground Truth: {e}")
#         print(traceback.format_exc())
#         return None

# main.py
from TorchSegmenter import TorchSegmenter
from SklearnSegmenter import SklearnSegmenter
from OpenCVSegmenter import OpenCVSegmenter
from SegmentationMetrics import SegmentationMetrics
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import cv2
from PIL import Image

# main.py - исправленная часть класса TorchImplementationValidator

class TorchImplementationValidator:
    """
    Класс для валидации кастомных PyTorch реализаций
    против оригинальных реализаций из библиотек
    """
    
    def __init__(self, output_dir: str = "./validation_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.validation_results = {}
        
        # Методы для валидации (только пороговые и операторные)
        self.threshold_methods = [
            "global_thresholding",
            "otsu_thresholding",
            "adaptive_thresholding",
            "threshold_niblack",
            "threshold_sauvola"
        ]
        
        self.edge_methods = [
            "sobel_edge",
            "canny_edge"
        ]

        self.region_methods = [
            "region_growing",
            "split_and_merge",
            "floodfill"
        ]

        self.clastering_methods = [
            "kmeans_segmentation",
            "dbscan_segmentation",
            "meanshift"
        ]

        self.active_contour_methods = [
            "active_contour",
            "gvf_contour",
            "morphological_snakes",
            "chan_vese",
        ]

        self.watershed_methods = [
            "watershed",
            "random_walker",
        ]

        self.super_pixel_methods = [
            # "quickshift",
            "slic",
            "felzenszwalb",
        ]

        self.interactive_methods = [
            "grabcut"
        ]
        
        # Пороги успешности валидации
        self.success_thresholds = {
            'iou': 0.85,      # IoU > 0.85 считается хорошим соответствием
            'dice': 0.90,     # Dice > 0.90
            'pixel_accuracy': 0.95  # Pixel Accuracy > 0.95
        }
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """
        Универсальная загрузка изображения для всех сегментаторов.
        Возвращает numpy array в формате RGB.
        """
        if isinstance(image_path, str) and os.path.exists(image_path):
            # Загружаем через PIL для единообразия
            img = Image.open(image_path).convert('RGB')
            return np.array(img)
        elif isinstance(image_path, np.ndarray):
            return image_path
        elif isinstance(image_path, Image.Image):
            return np.array(image_path.convert('RGB'))
        else:
            raise ValueError(f"Неподдерживаемый тип изображения: {type(image_path)}")
    
    def validate_threshold_methods(
        self,
        image_path: str,
        reference: str = "sklearn"  # "sklearn" или "opencv"
    ) -> Dict:
        """
        Валидация пороговых методов
        """
        print(f"\n{'='*60}")
        print(f"ВАЛИДАЦИЯ ПОРОГОВЫХ МЕТОДОВ")
        print(f"Референс: {reference.upper()}")
        print(f"{'='*60}")
        
        results = {}
        
        # Загружаем изображение ОДИН РАЗ
        img_array = self._load_image(image_path)
        
        for method in self.threshold_methods:
            print(f"\n📊 Метод: {method}")
            
            try:
                # 1. Torch реализация - передаём numpy array
                torch_segmenter = TorchSegmenter(method=method)
                torch_mask = torch_segmenter.segment(img_array)  # ✅ Передаём array, не путь!
                
                # 2. Референсная реализация
                if reference == "sklearn":
                    ref_segmenter = SklearnSegmenter(method=method)
                else:  # opencv
                    ref_segmenter = OpenCVSegmenter(method=method)
                
                ref_mask = ref_segmenter.segment(img_array)  # ✅ Передаём array, не путь!
                
                # 3. Вычисляем метрики соответствия
                metrics = SegmentationMetrics.calculate_all_metrics(
                    torch_mask, ref_mask, threshold=0.5, include_hausdorff=False
                )
                
                # 4. Определяем статус валидации
                validation_status = self._check_validation_status(metrics)
                
                results[method] = {
                    'torch_mask': torch_mask,
                    'reference_mask': ref_mask,
                    'metrics': metrics,
                    'validation_status': validation_status,
                    'success': True,
                    'reference_library': reference
                }
                
                # Вывод результатов
                status_icon = "✅" if validation_status == "PASS" else "⚠️"
                print(f"   {status_icon} IoU: {metrics['iou']:.4f}")
                print(f"   {status_icon} Dice: {metrics['dice']:.4f}")
                print(f"   {status_icon} Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
                print(f"   Статус: {validation_status}")
                
            except Exception as e:
                print(f"   ❌ Ошибка: {e}")
                import traceback
                traceback.print_exc()
                results[method] = {
                    'success': False,
                    'error': str(e),
                    'reference_library': reference
                }
        
        # Сохраняем результаты
        self._save_validation_results(results, "threshold_validation", reference)
        self._visualize_validation(results, img_array, "threshold", reference)
        
        return results
    
    def validate_edge_methods(
        self,
        image_path: str,
        reference: str = "opencv"  # Для edge методов лучше OpenCV
    ) -> Dict:
        """
        Валидация операторов границ
        """
        print(f"\n{'='*60}")
        print(f"ВАЛИДАЦИЯ ОПЕРАТОРОВ ГРАНИЦ")
        print(f"Референс: {reference.upper()}")
        print(f"{'='*60}")
        
        results = {}
        
        # Загружаем изображение ОДИН РАЗ
        img_array = self._load_image(image_path)
        
        for method in self.edge_methods:
            print(f"\n📊 Метод: {method}")
            
            try:
                # 1. Torch реализация
                torch_segmenter = TorchSegmenter(method=method)
                torch_mask = torch_segmenter.segment(img_array)
                
                # 2. Референсная реализация
                if reference == "sklearn":
                    ref_segmenter = SklearnSegmenter(method=method)
                else:  # opencv
                    ref_segmenter = OpenCVSegmenter(method=method)
                
                ref_mask = ref_segmenter.segment(img_array)
                
                # 3. Вычисляем метрики соответствия
                metrics = SegmentationMetrics.calculate_all_metrics(
                    torch_mask, ref_mask, threshold=0.5, include_hausdorff=False
                )
                
                # 4. Определяем статус валидации
                validation_status = self._check_validation_status(metrics)
                
                results[method] = {
                    'torch_mask': torch_mask,
                    'reference_mask': ref_mask,
                    'metrics': metrics,
                    'validation_status': validation_status,
                    'success': True,
                    'reference_library': reference
                }
                
                # Вывод результатов
                status_icon = "✅" if validation_status == "PASS" else "⚠️"
                print(f"   {status_icon} IoU: {metrics['iou']:.4f}")
                print(f"   {status_icon} Dice: {metrics['dice']:.4f}")
                print(f"   {status_icon} Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
                print(f"   Статус: {validation_status}")
                
            except Exception as e:
                print(f"   ❌ Ошибка: {e}")
                import traceback
                traceback.print_exc()
                results[method] = {
                    'success': False,
                    'error': str(e),
                    'reference_library': reference
                }
        
        # Сохраняем результаты
        self._save_validation_results(results, "edge_validation", reference)
        self._visualize_validation(results, img_array, "edge", reference)
        
        return results

    def validate_edge_methods_enhanced(
        self,
        image_path: str,
        reference: str = "opencv"  # Для edge методов лучше OpenCV
    ) -> Dict:
        """
        Валидация операторов границ
        """
        print(f"\n{'='*60}")
        print(f"ВАЛИДАЦИЯ ОПЕРАТОРОВ ГРАНИЦ")
        print(f"Референс: {reference.upper()}")
        print(f"{'='*60}")
        
        results = {}
        
        # Загружаем изображение ОДИН РАЗ
        img_array = self._load_image(image_path)
        
        for method in self.edge_methods:
            print(f"\n📊 Метод: {method}")
            
            try:
                # 1. Torch реализация
                torch_segmenter = OpenCVSegmenter(method=method)
                torch_mask = torch_segmenter.segment(img_array)
                
                # 2. Референсная реализация
                ref_segmenter = SklearnSegmenter(method=method)
                
                ref_mask = ref_segmenter.segment(img_array)
                
                # 3. Вычисляем метрики соответствия
                metrics = SegmentationMetrics.calculate_all_metrics(
                    torch_mask, ref_mask, threshold=0.5, include_hausdorff=False
                )
                
                # 4. Определяем статус валидации
                validation_status = self._check_validation_status(metrics)
                
                results[method] = {
                    'torch_mask': torch_mask,
                    'reference_mask': ref_mask,
                    'metrics': metrics,
                    'validation_status': validation_status,
                    'success': True,
                    'reference_library': reference
                }
                
                # Вывод результатов
                status_icon = "✅" if validation_status == "PASS" else "⚠️"
                print(f"   {status_icon} IoU: {metrics['iou']:.4f}")
                print(f"   {status_icon} Dice: {metrics['dice']:.4f}")
                print(f"   {status_icon} Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
                print(f"   Статус: {validation_status}")
                
            except Exception as e:
                print(f"   ❌ Ошибка: {e}")
                import traceback
                traceback.print_exc()
                results[method] = {
                    'success': False,
                    'error': str(e),
                    'reference_library': reference
                }
        
        # Сохраняем результаты
        self._save_validation_results(results, "edge_validation", reference)
        self._visualize_validation_enhanced(results, img_array, "edge", reference)
        
        return results
    
    def validate_region_methods(
        self,
        image_path: str,
        reference: str = "sklearn"  # "sklearn" или "opencv"
    ) -> Dict:
        """
        Валидация региональных методов
        """
        print(f"\n{'='*60}")
        print(f"ВАЛИДАЦИЯ РЕГИОНАЛЬНЫХ МЕТОДОВ")
        print(f"Референс: {reference.upper()}")
        print(f"{'='*60}")
        
        results = {}
        
        # Загружаем изображение ОДИН РАЗ
        img_array = self._load_image(image_path)
        
        for method in self.region_methods:
            print(f"\n📊 Метод: {method}")
            
            try:
                # 1. Torch реализация - передаём numpy array
                torch_segmenter = TorchSegmenter(method=method)
                torch_mask = torch_segmenter.segment(img_array)  # ✅ Передаём array, не путь!
                
                # 2. Референсная реализация
                if reference == "sklearn":
                    ref_segmenter = SklearnSegmenter(method=method)
                else:  # opencv
                    ref_segmenter = OpenCVSegmenter(method=method)
                
                ref_mask = ref_segmenter.segment(img_array)  # ✅ Передаём array, не путь!
                
                # 3. Вычисляем метрики соответствия
                metrics = SegmentationMetrics.calculate_all_metrics(
                    torch_mask, ref_mask, threshold=0.5, include_hausdorff=False
                )
                
                # 4. Определяем статус валидации
                validation_status = self._check_validation_status(metrics)
                
                results[method] = {
                    'torch_mask': torch_mask,
                    'reference_mask': ref_mask,
                    'metrics': metrics,
                    'validation_status': validation_status,
                    'success': True,
                    'reference_library': reference
                }
                
                # Вывод результатов
                status_icon = "✅" if validation_status == "PASS" else "⚠️"
                print(f"   {status_icon} IoU: {metrics['iou']:.4f}")
                print(f"   {status_icon} Dice: {metrics['dice']:.4f}")
                print(f"   {status_icon} Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
                print(f"   Статус: {validation_status}")
                
            except Exception as e:
                print(f"   ❌ Ошибка: {e}")
                import traceback
                traceback.print_exc()
                results[method] = {
                    'success': False,
                    'error': str(e),
                    'reference_library': reference
                }
        
        # Сохраняем результаты
        self._save_validation_results(results, "region_validation", reference)
        self._visualize_validation(results, img_array, "region", reference)
        
        return results
    
    def validate_interactive_methods(
        self,
        image_path: str,
        reference: str = "sklearn"  # "sklearn" или "opencv"
    ) -> Dict:
        """
        Валидация интерактивных методов
        """
        print(f"\n{'='*60}")
        print(f"ВАЛИДАЦИЯ ИНТЕРАКТИВНЫХ МЕТОДОВ")
        print(f"Референс: {reference.upper()}")
        print(f"{'='*60}")
        
        results = {}
        
        # Загружаем изображение ОДИН РАЗ
        img_array = self._load_image(image_path)
        
        for method in self.interactive_methods:
            print(f"\n📊 Метод: {method}")
            
            try:
                # 1. Torch реализация - передаём numpy array
                torch_segmenter = TorchSegmenter(method=method)
                torch_mask = torch_segmenter.segment(img_array)  # ✅ Передаём array, не путь!
                
                # 2. Референсная реализация
                if reference == "sklearn":
                    ref_segmenter = SklearnSegmenter(method=method)
                else:  # opencv
                    ref_segmenter = OpenCVSegmenter(method=method)
                
                ref_mask = ref_segmenter.segment(img_array)  # ✅ Передаём array, не путь!
                
                # 3. Вычисляем метрики соответствия
                metrics = SegmentationMetrics.calculate_all_metrics(
                    torch_mask, ref_mask, threshold=0.5, include_hausdorff=False
                )
                
                # 4. Определяем статус валидации
                validation_status = self._check_validation_status(metrics)
                
                results[method] = {
                    'torch_mask': torch_mask,
                    'reference_mask': ref_mask,
                    'metrics': metrics,
                    'validation_status': validation_status,
                    'success': True,
                    'reference_library': reference
                }
                
                # Вывод результатов
                status_icon = "✅" if validation_status == "PASS" else "⚠️"
                print(f"   {status_icon} IoU: {metrics['iou']:.4f}")
                print(f"   {status_icon} Dice: {metrics['dice']:.4f}")
                print(f"   {status_icon} Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
                print(f"   Статус: {validation_status}")
                
            except Exception as e:
                print(f"   ❌ Ошибка: {e}")
                import traceback
                traceback.print_exc()
                results[method] = {
                    'success': False,
                    'error': str(e),
                    'reference_library': reference
                }
        
        # Сохраняем результаты
        self._save_validation_results(results, "interactive_validation", reference)
        self._visualize_validation(results, img_array, "interactive", reference)
        
        return results
    
    def validate_clastering_methods(
        self,
        image_path: str,
        reference: str = "opencv"  # Для edge методов лучше OpenCV
    ) -> Dict:
        """
        Валидация методов кластеризации
        """
        print(f"\n{'='*60}")
        print(f"ВАЛИДАЦИЯ МЕТОДОВ КЛАСТЕРИЗАЦИИ")
        print(f"Референс: {reference.upper()}")
        print(f"{'='*60}")
        
        results = {}
        
        # Загружаем изображение ОДИН РАЗ
        img_array = self._load_image(image_path)
        
        for method in self.clastering_methods:
            print(f"\n📊 Метод: {method}")
            
            try:
                # 1. Torch реализация
                torch_segmenter = TorchSegmenter(method=method)
                torch_mask = torch_segmenter.segment(img_array)
                
                # 2. Референсная реализация
                if reference == "sklearn":
                    ref_segmenter = SklearnSegmenter(method=method)
                else:  # opencv
                    ref_segmenter = OpenCVSegmenter(method=method)
                
                ref_mask = ref_segmenter.segment(img_array)
                
                # 3. Вычисляем метрики соответствия
                metrics = SegmentationMetrics.calculate_all_metrics(
                    torch_mask, ref_mask, threshold=0.5, include_hausdorff=False
                )
                
                # 4. Определяем статус валидации
                validation_status = self._check_validation_status(metrics)
                
                results[method] = {
                    'torch_mask': torch_mask,
                    'reference_mask': ref_mask,
                    'metrics': metrics,
                    'validation_status': validation_status,
                    'success': True,
                    'reference_library': reference
                }
                
                # Вывод результатов
                status_icon = "✅" if validation_status == "PASS" else "⚠️"
                print(f"   {status_icon} IoU: {metrics['iou']:.4f}")
                print(f"   {status_icon} Dice: {metrics['dice']:.4f}")
                print(f"   {status_icon} Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
                print(f"   Статус: {validation_status}")
                
            except Exception as e:
                print(f"   ❌ Ошибка: {e}")
                import traceback
                traceback.print_exc()
                results[method] = {
                    'success': False,
                    'error': str(e),
                    'reference_library': reference
                }
        
        # Сохраняем результаты
        self._save_validation_results(results, "claster_validation", reference)
        self._visualize_validation(results, img_array, "claster", reference)
        
        return results
    
    def validate_active_contour_methods(
        self,
        image_path: str,
        reference: str = "opencv"  # Для edge методов лучше OpenCV
    ) -> Dict:
        """
        Валидация методов активных контуров
        """
        print(f"\n{'='*60}")
        print(f"ВАЛИДАЦИЯ МЕТОДОВ АКТИВНЫХ КОНТУРОВ")
        print(f"Референс: {reference.upper()}")
        print(f"{'='*60}")
        
        results = {}
        
        # Загружаем изображение ОДИН РАЗ
        img_array = self._load_image(image_path)
        
        for method in self.active_contour_methods:
            print(f"\n📊 Метод: {method}")
            
            try:
                # 1. Torch реализация
                torch_segmenter = TorchSegmenter(method=method)
                torch_mask = torch_segmenter.segment(img_array)
                
                # 2. Референсная реализация
                if reference == "sklearn":
                    ref_segmenter = SklearnSegmenter(method=method)
                else:  # opencv
                    ref_segmenter = OpenCVSegmenter(method=method)
                
                ref_mask = ref_segmenter.segment(img_array)
                
                # 3. Вычисляем метрики соответствия
                metrics = SegmentationMetrics.calculate_all_metrics(
                    torch_mask, ref_mask, threshold=0.5, include_hausdorff=False
                )
                
                # 4. Определяем статус валидации
                validation_status = self._check_validation_status(metrics)
                
                results[method] = {
                    'torch_mask': torch_mask,
                    'reference_mask': ref_mask,
                    'metrics': metrics,
                    'validation_status': validation_status,
                    'success': True,
                    'reference_library': reference
                }
                
                # Вывод результатов
                status_icon = "✅" if validation_status == "PASS" else "⚠️"
                print(f"   {status_icon} IoU: {metrics['iou']:.4f}")
                print(f"   {status_icon} Dice: {metrics['dice']:.4f}")
                print(f"   {status_icon} Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
                print(f"   Статус: {validation_status}")
                
            except Exception as e:
                print(f"   ❌ Ошибка: {e}")
                import traceback
                traceback.print_exc()
                results[method] = {
                    'success': False,
                    'error': str(e),
                    'reference_library': reference
                }
        
        # Сохраняем результаты
        self._save_validation_results(results, "active_contour_validation", reference)
        self._visualize_validation(results, img_array, "active_contour", reference)
        
        return results
    
    def validate_watershed_methods(
        self,
        image_path: str,
        reference: str = "opencv"  # Для edge методов лучше OpenCV
    ) -> Dict:
        """
        Валидация методов водораздела
        """
        print(f"\n{'='*60}")
        print(f"ВАЛИДАЦИЯ МЕТОДОВ ВОДОРАЗДЕЛА")
        print(f"Референс: {reference.upper()}")
        print(f"{'='*60}")
        
        results = {}
        
        # Загружаем изображение ОДИН РАЗ
        img_array = self._load_image(image_path)
        
        for method in self.watershed_methods:
            print(f"\n📊 Метод: {method}")
            
            try:
                # 1. Torch реализация
                torch_segmenter = TorchSegmenter(method=method)
                torch_mask = torch_segmenter.segment(img_array)
                
                # 2. Референсная реализация
                if reference == "sklearn":
                    ref_segmenter = SklearnSegmenter(method=method)
                else:  # opencv
                    ref_segmenter = OpenCVSegmenter(method=method)
                
                ref_mask = ref_segmenter.segment(img_array)
                
                # 3. Вычисляем метрики соответствия
                metrics = SegmentationMetrics.calculate_all_metrics(
                    torch_mask, ref_mask, threshold=0.5, include_hausdorff=False
                )
                
                # 4. Определяем статус валидации
                validation_status = self._check_validation_status(metrics)
                
                results[method] = {
                    'torch_mask': torch_mask,
                    'reference_mask': ref_mask,
                    'metrics': metrics,
                    'validation_status': validation_status,
                    'success': True,
                    'reference_library': reference
                }
                
                # Вывод результатов
                status_icon = "✅" if validation_status == "PASS" else "⚠️"
                print(f"   {status_icon} IoU: {metrics['iou']:.4f}")
                print(f"   {status_icon} Dice: {metrics['dice']:.4f}")
                print(f"   {status_icon} Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
                print(f"   Статус: {validation_status}")
                
            except Exception as e:
                print(f"   ❌ Ошибка: {e}")
                import traceback
                traceback.print_exc()
                results[method] = {
                    'success': False,
                    'error': str(e),
                    'reference_library': reference
                }
        
        # Сохраняем результаты
        self._save_validation_results(results, "watershed_validation", reference)
        self._visualize_validation(results, img_array, "watershed", reference)
        
        return results
    
    def validate_super_pixel_methods(
        self,
        image_path: str,
        reference: str = "opencv"  # Для edge методов лучше OpenCV
    ) -> Dict:
        """
        Валидация суперпиксельных методов
        """
        print(f"\n{'='*60}")
        print(f"ВАЛИДАЦИЯ СУПЕРПИКСЕЛЬНЫХ МЕТОДОВ")
        print(f"Референс: {reference.upper()}")
        print(f"{'='*60}")
        
        results = {}
        
        # Загружаем изображение ОДИН РАЗ
        img_array = self._load_image(image_path)
        
        for method in self.super_pixel_methods:
            print(f"\n📊 Метод: {method}")
            
            try:
                # 1. Torch реализация
                torch_segmenter = TorchSegmenter(method=method)
                torch_mask = torch_segmenter.segment(img_array)
                
                # 2. Референсная реализация
                if reference == "sklearn":
                    ref_segmenter = SklearnSegmenter(method=method)
                else:  # opencv
                    ref_segmenter = OpenCVSegmenter(method=method)
                
                ref_mask = ref_segmenter.segment(img_array)
                
                # 3. Вычисляем метрики соответствия
                metrics = SegmentationMetrics.calculate_all_metrics(
                    torch_mask, ref_mask, threshold=0.5, include_hausdorff=False
                )
                
                # 4. Определяем статус валидации
                validation_status = self._check_validation_status(metrics)
                
                results[method] = {
                    'torch_mask': torch_mask,
                    'reference_mask': ref_mask,
                    'metrics': metrics,
                    'validation_status': validation_status,
                    'success': True,
                    'reference_library': reference
                }
                
                # Вывод результатов
                status_icon = "✅" if validation_status == "PASS" else "⚠️"
                print(f"   {status_icon} IoU: {metrics['iou']:.4f}")
                print(f"   {status_icon} Dice: {metrics['dice']:.4f}")
                print(f"   {status_icon} Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
                print(f"   Статус: {validation_status}")
                
            except Exception as e:
                print(f"   ❌ Ошибка: {e}")
                import traceback
                traceback.print_exc()
                results[method] = {
                    'success': False,
                    'error': str(e),
                    'reference_library': reference
                }
        
        # Сохраняем результаты
        self._save_validation_results(results, "super_pixel_validation", reference)
        self._visualize_validation(results, img_array, "super_pixel", reference)
        
        return results
    
    def _check_validation_status(self, metrics: Dict) -> str:
        """Определяет статус валидации на основе метрик"""
        passed = 0
        total = 3
        
        if metrics['iou'] >= self.success_thresholds['iou']:
            passed += 1
        if metrics['dice'] >= self.success_thresholds['dice']:
            passed += 1
        if metrics['pixel_accuracy'] >= self.success_thresholds['pixel_accuracy']:
            passed += 1
        
        if passed == total:
            return "PASS"
        elif passed >= total // 2:
            return "WARNING"
        else:
            return "FAIL"
    
    def _save_validation_results(self, results: Dict, prefix: str, reference: str):
        """Сохранение результатов валидации"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(self.output_dir, f"{prefix}_{reference}_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)
        
        # Сохраняем маски и метрики
        for method, data in results.items():
            if data.get('success'):
                method_dir = os.path.join(results_dir, method)
                os.makedirs(method_dir, exist_ok=True)
                
                # Torch маска
                torch_mask = data['torch_mask']
                np.save(os.path.join(method_dir, "torch_mask.npy"), torch_mask)
                
                # Reference маска
                ref_mask = data['reference_mask']
                np.save(os.path.join(method_dir, "reference_mask.npy"), ref_mask)
                
                # Метрики
                metrics = data['metrics']
                metrics_path = os.path.join(method_dir, "metrics.txt")
                with open(metrics_path, 'w', encoding='utf-8') as f:
                    f.write(f"Результаты валидации: {method}\n")
                    f.write(f"Референсная библиотека: {reference}\n")
                    f.write(f"Статус: {data['validation_status']}\n")
                    f.write("="*50 + "\n")
                    for key, value in metrics.items():
                        if isinstance(value, float):
                            f.write(f"{key}: {value:.6f}\n")
                        else:
                            f.write(f"{key}: {value}\n")
        
        print(f"\n💾 Результаты сохранены: {results_dir}")
    
    def _visualize_validation(
        self,
        results: Dict,
        image_array: np.ndarray,  # ✅ Теперь принимаем array, не путь!
        validation_type: str,
        reference: str
    ):
        """Визуализация результатов валидации"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        original = image_array  # Уже загружено
        
        n_methods = len([r for r in results.values() if r.get('success')])
        if n_methods == 0:
            print("⚠️ Нет успешных результатов для визуализации")
            return
        
        fig, axes = plt.subplots(n_methods, 4, figsize=(20, 5*n_methods))
        if n_methods == 1:
            axes = axes.reshape(1, -1)
        
        row = 0
        for method, data in results.items():
            if not data.get('success'):
                continue
            
            torch_mask = data['torch_mask']
            ref_mask = data['reference_mask']
            metrics = data['metrics']
            status = data['validation_status']
            
            # Оригинальное изображение
            axes[row, 0].imshow(original)
            axes[row, 0].set_title(f"Original Image")
            axes[row, 0].axis('off')
            
            # Torch маска
            axes[row, 1].imshow(torch_mask, cmap='gray')
            axes[row, 1].set_title(f"Torch {method}\nIoU: {metrics['iou']:.3f}")
            axes[row, 1].axis('off')
            
            # Reference маска
            axes[row, 2].imshow(ref_mask, cmap='gray')
            axes[row, 2].set_title(f"{reference.upper()} {method}")
            axes[row, 2].axis('off')
            
            # Разность
            diff = np.abs(torch_mask.astype(float) - ref_mask.astype(float))
            im = axes[row, 3].imshow(diff, cmap='hot')
            status_color = 'green' if status == 'PASS' else 'orange' if status == 'WARNING' else 'red'
            axes[row, 3].set_title(f"Difference\nStatus: {status}", color=status_color)
            axes[row, 3].axis('off')
            plt.colorbar(im, ax=axes[row, 3], fraction=0.046)
            
            row += 1
        
        plt.suptitle(f"{validation_type.title()} Validation (Torch vs {reference.upper()})", fontsize=16)
        plt.tight_layout()
        
        viz_path = os.path.join(
            self.output_dir,
            f"{validation_type}_validation_{reference}_{timestamp}.jpg"
        )
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Визуализация: {viz_path}")

    def _visualize_validation_enhanced(
        self,
        results: Dict,
        image_array: np.ndarray,  # ✅ Теперь принимаем array, не путь!
        validation_type: str,
        reference: str
    ):
        """Визуализация результатов валидации"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        original = image_array  # Уже загружено
        
        n_methods = len([r for r in results.values() if r.get('success')])
        if n_methods == 0:
            print("⚠️ Нет успешных результатов для визуализации")
            return
        
        fig, axes = plt.subplots(n_methods, 4, figsize=(20, 5*n_methods))
        if n_methods == 1:
            axes = axes.reshape(1, -1)
        
        row = 0
        for method, data in results.items():
            if not data.get('success'):
                continue
            
            torch_mask = data['torch_mask']
            ref_mask = data['reference_mask']
            metrics = data['metrics']
            status = data['validation_status']
            
            # Оригинальное изображение
            axes[row, 0].imshow(original)
            axes[row, 0].set_title(f"Original Image")
            axes[row, 0].axis('off')
            
            # Torch маска
            axes[row, 1].imshow(torch_mask, cmap='gray')
            axes[row, 1].set_title(f"OpenCV {method}\nIoU: {metrics['iou']:.3f}")
            axes[row, 1].axis('off')
            
            # Reference маска
            axes[row, 2].imshow(ref_mask, cmap='gray')
            axes[row, 2].set_title(f"{reference.upper()} {method}")
            axes[row, 2].axis('off')
            
            # Разность
            diff = np.abs(torch_mask.astype(float) - ref_mask.astype(float))
            im = axes[row, 3].imshow(diff, cmap='hot')
            status_color = 'green' if status == 'PASS' else 'orange' if status == 'WARNING' else 'red'
            axes[row, 3].set_title(f"Difference\nStatus: {status}", color=status_color)
            axes[row, 3].axis('off')
            plt.colorbar(im, ax=axes[row, 3], fraction=0.046)
            
            row += 1
        
        plt.suptitle(f"{validation_type.title()} Validation (OpenCV vs {reference.upper()})", fontsize=16)
        plt.tight_layout()
        
        viz_path = os.path.join(
            self.output_dir,
            f"{validation_type}_validation_{reference}_{timestamp}.jpg"
        )
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Визуализация: {viz_path}")
    
    def generate_validation_report(self, all_results: Dict) -> str:
        """Генерация сводного отчёта по валидации"""
        report_lines = []
        report_lines.append("="*60)
        report_lines.append("ОТЧЁТ ПО ВАЛИДАЦИИ TORCH РЕАЛИЗАЦИЙ")
        report_lines.append("="*60)
        report_lines.append(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        total_methods = 0
        passed_methods = 0
        warning_methods = 0
        failed_methods = 0
        
        for validation_type, results in all_results.items():
            report_lines.append(f"\n{validation_type.upper()}")
            report_lines.append("-"*40)
            
            for method, data in results.items():
                if not data.get('success'):
                    continue
                
                total_methods += 1
                status = data['validation_status']
                
                if status == "PASS":
                    passed_methods += 1
                    icon = "✅"
                elif status == "WARNING":
                    warning_methods += 1
                    icon = "⚠️"
                else:
                    failed_methods += 1
                    icon = "❌"
                
                metrics = data['metrics']
                report_lines.append(
                    f"{icon} {method}: "
                    f"IoU={metrics['iou']:.3f}, "
                    f"Dice={metrics['dice']:.3f}, "
                    f"Acc={metrics['pixel_accuracy']:.3f} "
                    f"[{status}]"
                )
        
        report_lines.append("")
        report_lines.append("="*60)
        report_lines.append("СВОДНАЯ СТАТИСТИКА")
        report_lines.append("="*60)
        report_lines.append(f"Всего методов: {total_methods}")
        
        # ✅ ИСПРАВЛЕНИЕ: проверка на деление на ноль
        if total_methods > 0:
            report_lines.append(f"✅ PASS: {passed_methods} ({passed_methods/total_methods*100:.1f}%)")
            report_lines.append(f"⚠️ WARNING: {warning_methods} ({warning_methods/total_methods*100:.1f}%)")
            report_lines.append(f"❌ FAIL: {failed_methods} ({failed_methods/total_methods*100:.1f}%)")
        else:
            report_lines.append("⚠️ Нет данных для статистики (все методы не прошли)")
        
        report_lines.append("="*60)
        
        report = "\n".join(report_lines)
        
        # Сохраняем отчёт
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.output_dir, f"validation_report_{timestamp}.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n📄 Отчёт сохранён: {report_path}")
        print("\n" + report)
        
        return report


def main():
    """Основная функция запуска валидации"""
    print("="*60)
    print("ВАЛИДАЦИЯ TORCH РЕАЛИЗАЦИЙ МЕТОДОВ СЕГМЕНТАЦИИ")
    print("="*60)
    
    # Инициализация валидатора
    validator = TorchImplementationValidator(
        output_dir="./validation_results"
    )
    
    # Загрузка тестового изображения
    test_image_path = "./data/test_image_6.jpg"
    
    # Проверка наличия изображения
    if not os.path.exists(test_image_path):
        print(f"❌ Изображение не найдено: {test_image_path}")
        print("Загрузка тестового изображения...")
        
        try:
            import requests
            from io import BytesIO
            
            url = "https://i.pinimg.com/736x/17/e7/fc/1D7oZ9cqSef531ErnBAai8ZivwSPyqMCcs.jpg"
            response = requests.get(url)
            image = Image.open(BytesIO(response.content))
            image.save(test_image_path)
            print(f"✅ Изображение загружено: {test_image_path}")
            
        except Exception as e:
            print(f"❌ Ошибка загрузки: {e}")
            return
    
    all_results = {}
    
    # 1. Валидация пороговых методов (референс: scikit-learn/scikit-image)
    threshold_results_sklearn = validator.validate_threshold_methods(
        test_image_path,
        reference="sklearn"
    )
    all_results['threshold_sklearn'] = threshold_results_sklearn
    
    # 2. Валидация пороговых методов (референс: OpenCV)
    threshold_results_opencv = validator.validate_threshold_methods(
        test_image_path,
        reference="opencv"
    )
    all_results['threshold_opencv'] = threshold_results_opencv
    
    # 3. Валидация операторов границ (референс: OpenCV)
    edge_results_opencv = validator.validate_edge_methods(
        test_image_path,
        reference="opencv"
    )
    all_results['edge_opencv'] = edge_results_opencv
    
    # 4. Валидация операторов границ (референс: scikit-learn/scikit-image)
    edge_results_sklearn = validator.validate_edge_methods(
        test_image_path,
        reference="sklearn"
    )
    all_results['edge_sklearn'] = edge_results_sklearn

    # 4. Валидация операторов границ (референс: scikit-learn/scikit-image)
    edge_results_enhanced = validator.validate_edge_methods_enhanced(
        test_image_path,
        reference="sklearn"
    )
    all_results['edge_custom'] = edge_results_enhanced

    region_results_sklearn = validator.validate_region_methods(
        test_image_path,
        reference="sklearn"
    )
    all_results['region_sklearn'] = region_results_sklearn

    region_results_opencv = validator.validate_region_methods(
        test_image_path,
        reference="opencv"
    )
    all_results['region_opencv'] = region_results_opencv

    interactive_results_sklearn = validator.validate_interactive_methods(
        test_image_path,
        reference="sklearn"
    )
    all_results['interactive_sklearn'] = interactive_results_sklearn

    interactive_results_opencv = validator.validate_interactive_methods(
        test_image_path,
        reference="opencv"
    )
    all_results['interactive_opencv'] = interactive_results_opencv

    clastering_results_sklearn = validator.validate_clastering_methods(
        test_image_path,
        reference="sklearn"
    )
    all_results['clastering_sklearn'] = clastering_results_sklearn

    clastering_results_opencv = validator.validate_clastering_methods(
        test_image_path,
        reference="opencv"
    )
    all_results['clastering_opencv'] = clastering_results_opencv

    active_contour_results_sklearn = validator.validate_active_contour_methods(
        test_image_path,
        reference="sklearn"
    )
    all_results['active_contour_sklearn'] = active_contour_results_sklearn

    active_contour_results_opencv = validator.validate_active_contour_methods(
        test_image_path,
        reference="opencv"
    )
    all_results['active_contour_opencv'] = active_contour_results_opencv

    watershed_results_sklearn = validator.validate_watershed_methods(
        test_image_path,
        reference="sklearn"
    )
    all_results['watershed_sklearn'] = watershed_results_sklearn

    watershed_results_opencv = validator.validate_watershed_methods(
        test_image_path,
        reference="opencv"
    )
    all_results['watershed_opencv'] = watershed_results_opencv

    super_pixel_results_sklearn = validator.validate_super_pixel_methods(
        test_image_path,
        reference="sklearn"
    )
    all_results['super_pixel_sklearn'] = super_pixel_results_sklearn

    super_pixel_results_opencv = validator.validate_super_pixel_methods(
        test_image_path,
        reference="opencv"
    )
    all_results['super_pixel_opencv'] = super_pixel_results_opencv
    
    # 5. Генерация сводного отчёта
    validator.generate_validation_report(all_results)
    
    print(f"\n✅ Все результаты сохранены в: {validator.output_dir}")
    print("="*60)
    
    return validator, all_results


if __name__ == "__main__":
    validator, results = main()