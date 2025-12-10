from SegmentationTester import SegmentationTester
from torch_segmenter import TorchSegmenter
from cv2_sklearn_segmener import CV2SklearnSegmenter

def main():
    """Пример использования всех классов"""
    
    # Инициализация тестера
    tester = SegmentationTester()
    
    # Добавление методов CV2/Sklearn
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
    
    # Добавление методов PyTorch
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
    
    # Загрузка тестового изображения
    image_path = "test_image.jpg"
    
    # Сравнение всех методов
    print("Сравнение всех методов сегментации...")
    results = tester.compare_methods(image_path)
    
    # Визуализация сравнения
    tester.visualize_comparison(results, show_masks=True)
    
    # Бенчмарк производительности
    print("\nБенчмарк производительности методов...")
    df = tester.benchmark_methods(image_path, n_runs=3)
    
    # Сохранение результатов
    tester.save_results(results, "segmentation_comparison_results")
    
    return tester, results, df


if __name__ == "__main__":
    tester, results, df = main()