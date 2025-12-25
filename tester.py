# segmentation_comparator_extended.py
from SegmentationComparator import SegmentationComparator
import numpy as np
from typing import Union, Tuple, Dict, Any, List, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, jaccard_score
import itertools
import os
from datetime import datetime


class ExtendedSegmentationComparator(SegmentationComparator):
    """
    Расширенный компаратор с матричным сравнением всех методов.
    """
    
    def matrix_comparison(self,
                         image: np.ndarray,
                         methods_config: List[Dict[str, Any]],
                         reference_method: Optional[str] = None,
                         comparison_type: str = "all_vs_all",  # "all_vs_all", "all_vs_ref", "pairwise"
                         save_results: bool = True,
                         output_dir: str = "matrix_comparison_results") -> Dict[str, Any]:
        """
        Матричное сравнение всех методов между собой.
        
        Args:
            image: Входное изображение
            methods_config: Конфигурация методов
            reference_method: Референсный метод (если None - сравнение всех со всеми)
            comparison_type: Тип сравнения
                - "all_vs_all": все методы сравниваются со всеми (N x N матрица)
                - "all_vs_ref": все методы сравниваются с референсным
                - "pairwise": сравнение всех возможных пар (без дубликатов)
            save_results: Сохранять ли результаты
            output_dir: Директория для сохранения
        
        Returns:
            Dict[str, Any]: Результаты сравнения
        """
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(output_dir, f"comparison_{timestamp}")
            os.makedirs(output_dir, exist_ok=True)
        
        # Генерируем имена методов для удобства
        method_names = []
        method_objects = {}
        
        for config in methods_config:
            method_name = config.get('name')
            method_type = config.get('type', 'skimage')
            method_params = config.get('params', {})
            
            full_name = f"{method_type}_{method_name}"
            method_names.append(full_name)
            method_objects[full_name] = {
                'type': method_type,
                'name': method_name,
                'params': method_params
            }
        
        # Выполняем сегментацию всеми методами
        print(f"Выполняем сегментацию {len(method_names)} методами...")
        masks = {}
        execution_times = {}
        method_infos = {}
        
        for full_name in method_names:
            config = method_objects[full_name]
            
            try:
                if config['type'] == "skimage":
                    mask, info = self.segment_with_skimage(
                        image, config['name'], **config['params'])
                elif config['type'] == "sklearn":
                    mask, info = self.segment_with_sklearn(
                        image, config['name'], **config['params'])
                else:
                    continue
                
                masks[full_name] = mask
                execution_times[full_name] = info.get('execution_time', 0)
                method_infos[full_name] = info
                
                print(f"  ✅ {full_name}: {execution_times[full_name]:.3f}s")
                
            except Exception as e:
                print(f"  ❌ {full_name}: {e}")
                # Создаем пустую маску для методов с ошибкой
                if len(image.shape) == 3:
                    h, w = image.shape[:2]
                else:
                    h, w = image.shape
                masks[full_name] = np.zeros((h, w), dtype=np.uint8)
                execution_times[full_name] = 0
                method_infos[full_name] = {'error': str(e)}
        
        # Выбираем стратегию сравнения
        if comparison_type == "all_vs_ref" and reference_method:
            # Все методы сравниваем с референсным
            comparison_pairs = [(reference_method, other) for other in method_names 
                              if other != reference_method]
            ref_name = reference_method
        elif comparison_type == "pairwise":
            # Все возможные пары без дубликатов
            comparison_pairs = list(itertools.combinations(method_names, 2))
            ref_name = None
        else:  # "all_vs_all"
            # Полная матрица N x N (включая сравнение с самим собой)
            comparison_pairs = [(m1, m2) for m1 in method_names 
                              for m2 in method_names]
            ref_name = None
        
        # Выполняем сравнения
        print(f"\nВыполняем сравнение {len(comparison_pairs)} пар...")
        comparison_results = []
        
        for i, (method1, method2) in enumerate(comparison_pairs):
            if method1 not in masks or method2 not in masks:
                continue
            
            mask1 = masks[method1]
            mask2 = masks[method2]
            
            try:
                # Вычисляем метрики
                metrics = self.compute_metrics(mask1, mask2, method1, method2)
                
                result = {
                    'method1': method1,
                    'method2': method2,
                    **metrics,
                    'time1': execution_times.get(method1, 0),
                    'time2': execution_times.get(method2, 0),
                    'time_diff': abs(execution_times.get(method1, 0) - 
                                   execution_times.get(method2, 0))
                }
                
                comparison_results.append(result)
                
                # Прогресс
                if (i + 1) % 10 == 0:
                    print(f"  Обработано {i + 1}/{len(comparison_pairs)} пар...")
                    
            except Exception as e:
                print(f"  Ошибка сравнения {method1} vs {method2}: {e}")
        
        # Создаем DataFrame
        df_comparisons = pd.DataFrame(comparison_results)
        
        if save_results:
            # Сохраняем все маски
            masks_dir = os.path.join(output_dir, "masks")
            os.makedirs(masks_dir, exist_ok=True)
            
            for name, mask in masks.items():
                mask_path = os.path.join(masks_dir, f"{name}_mask.png")
                plt.imsave(mask_path, mask, cmap='gray')
            
            # Сохраняем все изображения
            images_dir = os.path.join(output_dir, "images")
            os.makedirs(images_dir, exist_ok=True)
            
            # Оригинал
            if len(image.shape) == 2:
                plt.imsave(os.path.join(images_dir, "original.png"), 
                          image, cmap='gray')
            else:
                plt.imsave(os.path.join(images_dir, "original.png"), image)
            
            # Наложения
            for name, mask in masks.items():
                if len(image.shape) == 2:
                    overlay = np.stack([image] * 3, axis=-1)
                else:
                    overlay = image.copy()
                
                overlay[mask > 127] = [255, 0, 0]  # Красный
                overlay_path = os.path.join(images_dir, f"{name}_overlay.png")
                plt.imsave(overlay_path, overlay)
            
            # Сохраняем результаты
            self._save_matrix_results(df_comparisons, masks, method_infos, 
                                     output_dir, comparison_type, ref_name)
        
        return {
            'df_comparisons': df_comparisons,
            'masks': masks,
            'execution_times': execution_times,
            'method_infos': method_infos
        }
    
    def _save_matrix_results(self,
                            df_comparisons: pd.DataFrame,
                            masks: Dict[str, np.ndarray],
                            method_infos: Dict[str, Any],
                            output_dir: str,
                            comparison_type: str,
                            reference_method: Optional[str] = None):
        """Сохраняет результаты матричного сравнения."""
        
        # 1. Сохраняем DataFrame
        csv_path = os.path.join(output_dir, "comparisons.csv")
        df_comparisons.to_csv(csv_path, index=False)
        print(f"📊 CSV с результатами: {csv_path}")
        
        # 2. Сводная таблица метрик
        summary_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'jaccard']
        
        if comparison_type == "all_vs_ref" and reference_method:
            # Средние метрики по сравнению с референсом
            ref_comparisons = df_comparisons[df_comparisons['method1'] == reference_method]
            if not ref_comparisons.empty:
                summary_df = ref_comparisons[['method2'] + summary_metrics].copy()
                summary_df = summary_df.rename(columns={'method2': 'method'})
                summary_df = summary_df.sort_values('f1_score', ascending=False)
                
                summary_path = os.path.join(output_dir, "summary_vs_ref.csv")
                summary_df.to_csv(summary_path, index=False)
                
                print(f"📋 Сводная таблица (vs {reference_method}): {summary_path}")
        
        # 3. Матрицы сравнения
        methods = sorted(list(masks.keys()))
        n_methods = len(methods)
        
        # Создаем матрицы для каждой метрики
        for metric in ['f1_score', 'accuracy', 'jaccard']:
            if metric not in df_comparisons.columns:
                continue
            
            # Создаем матрицу N x N
            matrix = np.zeros((n_methods, n_methods))
            
            for i, m1 in enumerate(methods):
                for j, m2 in enumerate(methods):
                    if i == j:
                        matrix[i, j] = 1.0  # Само с собой - идеальное совпадение
                    else:
                        # Ищем сравнение в DataFrame
                        mask = ((df_comparisons['method1'] == m1) & 
                               (df_comparisons['method2'] == m2)) | \
                               ((df_comparisons['method1'] == m2) & 
                               (df_comparisons['method2'] == m1))
                        
                        if mask.any():
                            matrix[i, j] = df_comparisons.loc[mask, metric].values[0]
                        else:
                            matrix[i, j] = np.nan
            
            # Визуализируем матрицу
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Сокращаем имена методов для подписей
            short_names = [name[:15] + "..." if len(name) > 15 else name 
                          for name in methods]
            
            im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1)
            ax.set_xticks(np.arange(n_methods))
            ax.set_yticks(np.arange(n_methods))
            ax.set_xticklabels(short_names, rotation=45, ha='right')
            ax.set_yticklabels(short_names)
            
            # Добавляем значения в ячейки
            for i in range(n_methods):
                for j in range(n_methods):
                    if not np.isnan(matrix[i, j]):
                        text = ax.text(j, i, f"{matrix[i, j]:.2f}",
                                     ha="center", va="center", 
                                     color="black" if matrix[i, j] < 0.7 else "white",
                                     fontsize=8)
            
            ax.set_title(f"Матрица сравнения: {metric.upper()}", fontsize=14)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
            
            matrix_path = os.path.join(output_dir, f"{metric}_matrix.png")
            plt.savefig(matrix_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"📈 Матрица {metric}: {matrix_path}")
        
        # 4. Визуализация всех масок
        self._visualize_all_masks(masks, output_dir)
        
        # 5. Создаем HTML отчет
        self._create_html_report(df_comparisons, masks, method_infos, 
                                output_dir, comparison_type, reference_method)
    
    def _visualize_all_masks(self,
                            masks: Dict[str, np.ndarray],
                            output_dir: str):
        """Визуализирует все маски в одной фигуре."""
        methods = list(masks.keys())
        n_methods = len(methods)
        
        # Определяем размер сетки
        n_cols = 4
        n_rows = (n_methods + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 5))
        axes = axes.flatten()
        
        for i, (name, mask) in enumerate(masks.items()):
            ax = axes[i]
            ax.imshow(mask, cmap='gray')
            ax.set_title(f"{name}", fontsize=10)
            ax.axis('off')
        
        # Скрываем пустые оси
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        
        plt.suptitle("Все маски сегментации", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        all_masks_path = os.path.join(output_dir, "all_masks.png")
        plt.savefig(all_masks_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"🖼️ Все маски: {all_masks_path}")
    
    def _create_html_report(self,
                           df_comparisons: pd.DataFrame,
                           masks: Dict[str, np.ndarray],
                           method_infos: Dict[str, Any],
                           output_dir: str,
                           comparison_type: str,
                           reference_method: Optional[str] = None):
        """Создает HTML отчет с результатами."""
        
        html_path = os.path.join(output_dir, "report.html")
        
        # Статистика по методам
        methods_stats = []
        for name, mask in masks.items():
            mask_binary = mask > 127
            area = np.sum(mask_binary)
            total_pixels = mask.size
            coverage = area / total_pixels * 100
            
            methods_stats.append({
                'method': name,
                'area': area,
                'coverage': f"{coverage:.1f}%",
                'pixels': f"{area:,}",
                'time': method_infos.get(name, {}).get('execution_time', 0)
            })
        
        # Топ методов по F1 (если есть референс)
        if reference_method and 'f1_score' in df_comparisons.columns:
            ref_df = df_comparisons[df_comparisons['method1'] == reference_method]
            if not ref_df.empty:
                top_methods = ref_df.nlargest(5, 'f1_score')[['method2', 'f1_score']]
                top_methods_html = top_methods.to_html(index=False, 
                                                      float_format=lambda x: f"{x:.3f}")
            else:
                top_methods_html = "<p>Нет данных</p>"
        else:
            top_methods_html = "<p>Сравнение всех со всеми</p>"
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Отчет сравнения методов сегментации</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2, h3 {{ color: #333; }}
                    .summary {{ background: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
                    .metric-card {{ background: white; padding: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                    table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    img {{ max-width: 100%; height: auto; margin: 10px 0; }}
                    .highlight {{ background-color: #e6f7ff; }}
                </style>
            </head>
            <body>
                <h1>📊 Отчет сравнения методов сегментации</h1>
                
                <div class="summary">
                    <h2>Общая информация</h2>
                    <p><strong>Дата:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p><strong>Всего методов:</strong> {len(masks)}</p>
                    <p><strong>Тип сравнения:</strong> {comparison_type}</p>
                    <p><strong>Референсный метод:</strong> {reference_method if reference_method else 'Нет (все со всеми)'}</p>
                </div>
                
                <h2>📈 Матрицы сравнения</h2>
                <div class="metrics">
                    <div class="metric-card">
                        <h3>F1-Score матрица</h3>
                        <img src="f1_score_matrix.png" alt="F1 Matrix">
                    </div>
                    <div class="metric-card">
                        <h3>Accuracy матрица</h3>
                        <img src="accuracy_matrix.png" alt="Accuracy Matrix">
                    </div>
                    <div class="metric-card">
                        <h3>Все маски</h3>
                        <img src="all_masks.png" alt="All Masks">
                    </div>
                </div>
                
                <h2>🏆 Топ методов</h2>
                {top_methods_html}
                
                <h2>📋 Статистика методов</h2>
                <table>
                    <tr>
                        <th>Метод</th>
                        <th>Площадь маски</th>
                        <th>Покрытие</th>
                        <th>Время (с)</th>
                    </tr>
            """)
            
            for stat in sorted(methods_stats, key=lambda x: x['area'], reverse=True):
                f.write(f"""
                    <tr>
                        <td>{stat['method']}</td>
                        <td>{stat['pixels']}</td>
                        <td>{stat['coverage']}</td>
                        <td>{stat['time']:.3f}</td>
                    </tr>
                """)
            
            f.write("""
                </table>
                
                <h2>🔗 Быстрые ссылки</h2>
                <ul>
                    <li><a href="comparisons.csv">CSV с результатами сравнения</a></li>
                    <li><a href="masks/">Папка с масками</a></li>
                    <li><a href="images/">Папка с изображениями</a></li>
                </ul>
                
                <footer>
                    <p>Сгенерировано автоматически с помощью SegmentationComparator</p>
                </footer>
            </body>
            </html>
            """)
        
        print(f"📄 HTML отчет: {html_path}")


# Пример использования с полным сравнением
def comprehensive_comparison_example():
    """Пример полного сравнения всех методов"""
    
    # Загрузка тестового изображения
    import cv2
    from PIL import Image
    import requests
    from io import BytesIO
    
    url = "https://i.pinimg.com/736x/17/66/c4/1766c4f667af39f91172ef8eb21ab18a.jpg"
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img_np = np.array(img)
    
    # Конфигурация ВСЕХ методов
    all_methods_config = [
        # sklearn методы
        {"name": "kmeans", "type": "sklearn", "params": {"n_clusters": 3}},
        {"name": "dbscan", "type": "sklearn", "params": {"eps": 0.5, "min_samples": 5}},
        {"name": "meanshift", "type": "sklearn", "params": {"bandwidth": 0.5}},
        {"name": "gmm", "type": "sklearn", "params": {"n_components": 3}},
        
        # skimage методы сегментации
        {"name": "felzenszwalb", "type": "skimage", "params": {"scale": 100, "sigma": 0.8}},
        {"name": "slic", "type": "skimage", "params": {"n_segments": 100}},
        {"name": "quickshift", "type": "skimage", "params": {"kernel_size": 3}},
        {"name": "watershed", "type": "skimage", "params": {}},
        {"name": "random_walker", "type": "skimage", "params": {"beta": 130}},
        {"name": "chan_vese", "type": "skimage", "params": {"max_iter": 100}},
        {"name": "active_contour", "type": "skimage", "params": {"max_iter": 100}},
        
        # skimage пороговые методы
        {"name": "threshold_otsu", "type": "skimage", "params": {}},
        {"name": "threshold_niblack", "type": "skimage", "params": {"window_size": 25}},
        {"name": "threshold_sauvola", "type": "skimage", "params": {"window_size": 25}},
        
        # skimage детекторы границ
        {"name": "sobel", "type": "skimage", "params": {"threshold": 0.1}},
        {"name": "canny", "type": "skimage", "params": {"sigma": 1.0}},
    ]
    
    # Создаем компаратор
    comparator = ExtendedSegmentationComparator()
    
    print("=" * 60)
    print("ПОЛНОЕ МАТРИЧНОЕ СРАВНЕНИЕ ВСЕХ МЕТОДОВ")
    print("=" * 60)
    
    # Вариант 1: Сравнение всех со всеми
    print("\n1. Сравнение всех методов со всеми...")
    results_all = comparator.matrix_comparison(
        img_np,
        methods_config=all_methods_config,
        comparison_type="all_vs_all",
        output_dir="all_vs_all_comparison"
    )
    
    # Вариант 2: Все методы vs референс (например, felzenszwalb)
    print("\n2. Все методы vs референс (felzenszwalb)...")
    results_vs_ref = comparator.matrix_comparison(
        img_np,
        methods_config=all_methods_config,
        reference_method="skimage_felzenszwalb",
        comparison_type="all_vs_ref",
        output_dir="all_vs_felzenszwalb"
    )
    
    # Вариант 3: Только попарное сравнение
    print("\n3. Попарное сравнение всех методов...")
    results_pairwise = comparator.matrix_comparison(
        img_np,
        methods_config=all_methods_config[:8],  # Берем первые 8 для скорости
        comparison_type="pairwise",
        output_dir="pairwise_comparison"
    )
    
    # Анализ результатов
    print("\n" + "=" * 60)
    print("АНАЛИЗ РЕЗУЛЬТАТОВ")
    print("=" * 60)
    
    if 'df_comparisons' in results_all:
        df_all = results_all['df_comparisons']
        
        # Находим наиболее похожие методы
        print("\nСамые похожие пары методов (F1 > 0.9):")
        high_similarity = df_all[df_all['f1_score'] > 0.9]
        
        if not high_similarity.empty:
            # Исключаем сравнение с самим собой
            high_similarity = high_similarity[high_similarity['method1'] != high_similarity['method2']]
            top_pairs = high_similarity.nlargest(10, 'f1_score')[['method1', 'method2', 'f1_score']]
            print(top_pairs.to_string(index=False))
        else:
            print("Нет пар с F1 > 0.9")
        
        # Находим наиболее разные методы
        print("\nСамые разные пары методов (F1 < 0.3):")
        low_similarity = df_all[df_all['f1_score'] < 0.3]
        
        if not low_similarity.empty:
            low_similarity = low_similarity[low_similarity['method1'] != low_similarity['method2']]
            bottom_pairs = low_similarity.nsmallest(10, 'f1_score')[['method1', 'method2', 'f1_score']]
            print(bottom_pairs.to_string(index=False))
    
    return comparator, results_all, results_vs_ref


# Функция для быстрого сравнения
def quick_comparison(image: np.ndarray, 
                    methods_list: List[str] = None,
                    output_dir: str = "quick_comparison"):
    """
    Быстрое сравнение популярных методов.
    
    Args:
        image: Входное изображение
        methods_list: Список методов для сравнения (если None - стандартный набор)
        output_dir: Директория для сохранения
    """
    if methods_list is None:
        methods_list = [
            "kmeans", "dbscan", "felzenszwalb", "slic",
            "threshold_otsu", "canny", "watershed"
        ]
    
    # Автоматически определяем тип метода
    methods_config = []
    sklearn_methods = ["kmeans", "dbscan", "meanshift", "gmm"]
    
    for method in methods_list:
        if method in sklearn_methods:
            methods_config.append({
                "name": method,
                "type": "sklearn",
                "params": {}
            })
        else:
            methods_config.append({
                "name": method,
                "type": "skimage",
                "params": {}
            })
    
    comparator = ExtendedSegmentationComparator()
    
    print(f"Быстрое сравнение {len(methods_list)} методов...")
    
    results = comparator.matrix_comparison(
        image,
        methods_config=methods_config,
        comparison_type="all_vs_all",
        output_dir=output_dir
    )
    
    return comparator, results


# Интеграция с вашим main.py
def integrate_with_your_pipeline():
    """
    Пример интеграции с вашим существующим пайплайном
    """
    
    # Предположим, у вас есть ваш тестер
    from SegmentationTester import SegmentationTester
    from cv2SklearnSegmenter import CV2SklearnSegmenter
    
    # Создаем ваш тестер
    your_tester = SegmentationTester()
    
    # Добавляем ваши методы
    your_tester.add_method("My_KMeans", CV2SklearnSegmenter("kmeans_segmentation", k=3))
    your_tester.add_method("My_Watershed", CV2SklearnSegmenter("watershed"))
    your_tester.add_method("My_Otsu", CV2SklearnSegmenter("otsu_thresholding"))
    
    # Загружаем изображение
    image_path = "test_image.jpg"
    image = cv2.imread(image_path)
    
    # Создаем компаратор
    comparator = ExtendedSegmentationComparator()
    
    # Конфигурация для сравнения
    comparison_config = [
        # Ваши методы
        {"name": "My_KMeans", "type": "custom", "segmenter": your_tester.methods["My_KMeans"]},
        {"name": "My_Watershed", "type": "custom", "segmenter": your_tester.methods["My_Watershed"]},
        {"name": "My_Otsu", "type": "custom", "segmenter": your_tester.methods["My_Otsu"]},
        
        # Референсные методы
        {"name": "kmeans", "type": "sklearn", "params": {"n_clusters": 3}},
        {"name": "watershed", "type": "skimage", "params": {}},
        {"name": "threshold_otsu", "type": "skimage", "params": {}},
    ]
    
    # Выполняем сегментацию
    masks = {}
    
    for config in comparison_config:
        if config["type"] == "custom":
            segmenter = config["segmenter"]
            mask, _ = segmenter.segment_with_mask(image)
            masks[config["name"]] = mask
        else:
            if config["type"] == "sklearn":
                mask, _ = comparator.segment_with_sklearn(image, config["name"], **config["params"])
            else:  # skimage
                mask, _ = comparator.segment_with_skimage(image, config["name"], **config["params"])
            masks[config["name"]] = mask
    
    # Сравниваем все со всеми
    methods = list(masks.keys())
    comparison_results = []
    
    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods[i+1:], i+1):
            metrics = comparator.compute_metrics(masks[m1], masks[m2], m1, m2)
            
            comparison_results.append({
                'method1': m1,
                'method2': m2,
                **metrics
            })
    
    df_comparison = pd.DataFrame(comparison_results)
    
    print("\nСравнение ваших методов с референсными:")
    print(df_comparison[['method1', 'method2', 'f1_score', 'accuracy']].to_string(index=False))
    
    return df_comparison


if __name__ == "__main__":
    # Запуск полного сравнения
    comparator, results_all, results_vs_ref = comprehensive_comparison_example()
    
    print("\n✅ Сравнение завершено!")
    print("Результаты сохранены в папках:")
    print("  - all_vs_all_comparison/")
    print("  - all_vs_felzenszwalb/")
    print("  - pairwise_comparison/")