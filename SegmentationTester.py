# SegmentationTester.py
from BaseSegmenter import BaseSegmenter
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from typing import Union, Tuple, List, Dict, Any, Optional
import time
import json
from datetime import datetime

class SegmentationTester:
    """Класс для тестирования и сравнения методов сегментации"""
    
    def __init__(self,
                 base_output_dir: str = "segmentation_results"
    ) -> None:
        self.methods = {}
        self.results = {}
        self.base_output_dir: str = base_output_dir
        self.current_test_id: str = None

    def _create_test_directory(self, 
                               test_name: str = None
    ) -> str:
        """Создает уникальную директорию для теста"""
        timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_dir: str
        if test_name:
            test_dir = f"{test_name}_{timestamp}"
        else:
            test_dir = f"test_{timestamp}"
        
        full_path: str = os.path.join(self.base_output_dir, test_dir)
        os.makedirs(full_path, exist_ok=True)
        
        # Создаем поддиректории
        os.makedirs(os.path.join(full_path, "images"), exist_ok=True)
        os.makedirs(os.path.join(full_path, "masks"), exist_ok=True)
        os.makedirs(os.path.join(full_path, "comparisons"), exist_ok=True)
        os.makedirs(os.path.join(full_path, "statistics"), exist_ok=True)
        
        self.current_test_id: str = test_dir
        print(f"📁 Создана директория для теста: {full_path}")
        return full_path
    
    def add_method(self, 
                   name: str, 
                   segmenter: BaseSegmenter
    ) -> None:
        """Добавление метода сегментации"""
        self.methods[name] = segmenter
    
    def test_single_method(self, 
                           image: Union[str, np.ndarray, Image.Image], 
                           method_name: str, 
                           save_path: Optional[str] = None,
                           output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Тестирование одного метода с сохранением результатов"""
        if method_name not in self.methods:
            raise ValueError(f"Метод {method_name} не найден")
        
        segmenter = self.methods[method_name]
        
        # Измеряем время выполнения
        start_time: float = time.time()
        result: np.ndarray
        mask: np.ndarray
        result, mask = segmenter.segment_with_mask(image)
        execution_time: float = time.time() - start_time

        # Получаем оригинальное изображение для сохранения
        original_img: Image.Image
        img_array: np.ndarray
        if isinstance(image, str):
            original_img = Image.open(image).convert('RGB')
            img_array = np.array(original_img)
        elif isinstance(image, Image.Image):
            original_img = image
            img_array = np.array(image)
        else:
            img_array = image
            original_img = Image.fromarray(image.astype(np.uint8))
        
        # Статистика
        mask_area = np.sum(mask > 0)
        total_pixels = mask.shape[0] * mask.shape[1]
        
        result_data: Dict[str, Any] = {
            'method': method_name,
            'result': result,
            'mask': mask,
            'time': execution_time,
            'mask_area': mask_area,
            'mask_percentage': (mask_area / total_pixels) * 100,
            'image_shape': result.shape,
            'timestamp': datetime.now().isoformat()
        }
        
        # Сохранение результатов
        if output_dir:
            # Сохраняем оригинальное изображение
            orig_path: str = os.path.join(output_dir, "images", "original.jpg")
            original_img.save(orig_path)
            
            # Сохраняем результат сегментации
            result_path: str = os.path.join(output_dir, "images", f"{method_name}_result.jpg")
            result_pil: Image.Image = Image.fromarray(result.astype(np.uint8))
            result_pil.save(result_path)
            
            # Сохраняем маску
            mask_path: str = os.path.join(output_dir, "masks", f"{method_name}_mask.png")
            mask_pil: Image.Image = Image.fromarray(mask.astype(np.uint8))
            mask_pil.save(mask_path)
            
            # Сохраняем наложение (overlay)
            try:
                # Создаем overlay (50% оригинал + 50% результат)
                overlay = img_array * 0.5 + result * 0.5
                overlay = overlay.astype(np.uint8)
                overlay_path: str = os.path.join(output_dir, "images", f"{method_name}_overlay.jpg")
                overlay_pil: Image.Image = Image.fromarray(overlay)
                overlay_pil.save(overlay_path)
                
                result_data['overlay_path'] = overlay_path
            except:
                pass
            
            result_data['result_path'] = result_path
            result_data['mask_path'] = mask_path
            result_data['original_path'] = orig_path
            
            print(f"✅ {method_name}: сохранено в {output_dir}")
        
        elif save_path:
            result_pil = Image.fromarray(result.astype(np.uint8))
            result_pil.save(save_path)
            print(f"✅ Результат сохранен: {save_path}")
        
        print(f"   ⏱️ Время: {execution_time:.2f}s, 📏 Площадь: {result_data['mask_percentage']:.1f}%")
        
        return result_data
    
    def compare_methods(self, 
                        image: Union[str, np.ndarray, Image.Image], 
                        method_names: List[str] = None, 
                        figsize: Tuple[int, int] = (20, 15),
                        save_comparison: bool = True,
                        test_name: str = None,
                        show_plots: bool = True
    ) -> Dict[str, Any]:
        """Сравнение нескольких методов"""
        if method_names is None:
            method_names: List[str] = list(self.methods.keys())
        
        test_dir: str = self._create_test_directory(test_name)
        results = {}

        original_img: Image.Image
        image_path: str
        # Оригинальное изображение
        if isinstance(image, str):
            original_img = Image.open(image).convert('RGB')
            image_path = image
        elif isinstance(image, Image.Image):
            original_img = image
            image_path = None
        else:
            original_img = Image.fromarray(image.astype(np.uint8))
            image_path = None

        # Сохраняем оригинальное изображение в директории теста
        orig_save_path: str = os.path.join(test_dir, "images", "original.jpg")
        original_img.save(orig_save_path)
        print(f"📸 Оригинальное изображение сохранено: {orig_save_path}")
        
        # Создаем фигуру для отображения
        n_methods: int = len(method_names)
        n_cols: int = min(4, n_methods + 1)  # +1 для оригинального изображения
        n_rows: int = (n_methods + n_cols) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()
        
        axes[0].imshow(original_img)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        all_stats = []
        # Тестируем каждый метод
        for i, method_name in enumerate(method_names, 1):
            if i >= len(axes):
                break
            
            try:
                result_data: Dict[str, Any] = self.test_single_method(
                    image, 
                    method_name, 
                    output_dir=test_dir
                )
                results[method_name] = result_data
                
                # Добавляем статистику
                stats: Dict[str, Any] = {
                    'method': method_name,
                    'time_seconds': result_data['time'],
                    'mask_area_pixels': result_data['mask_area'],
                    'mask_percentage': result_data['mask_percentage'],
                    'image_shape': result_data['image_shape']
                }
                all_stats.append(stats)
                
                # Отображение результата
                axes[i].imshow(result_data['result'])
                title: str = f"{method_name}\n{result_data['time']:.2f}s, {result_data['mask_percentage']:.1f}%"
                axes[i].set_title(title, fontsize=9)
                axes[i].axis('off')
                
                print(f"{method_name}: {result_data['time']:.2f}s, {result_data['mask_percentage']:.1f}% площади")
                
            except Exception as e:
                error_msg: str = str(e)[:50]
                axes[i].text(0.5, 0.5, f"Error:\n{error_msg}", 
                           ha='center', va='center', 
                           transform=axes[i].transAxes,
                           fontsize=8)
                axes[i].set_title(f"{method_name}\n(Error)", fontsize=9)
                axes[i].axis('off')
                
                print(f"❌ Ошибка в методе {method_name}: {e}")
                
                # Добавляем запись об ошибке в статистику
                stats: Dict[str, Any] = {
                    'method': method_name,
                    'error': error_msg,
                    'time_seconds': None,
                    'mask_area_pixels': None,
                    'mask_percentage': None
                }
                all_stats.append(stats)
        
        # Скрываем пустые оси
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        
        plt.suptitle(f"Сравнение методов сегментации\n{self.current_test_id}", 
                    fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Сохраняем сравнение
        if save_comparison:
            comparison_path: str = os.path.join(test_dir, "comparisons", "methods_comparison.jpg")
            plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
            print(f"📊 Сравнительный график сохранен: {comparison_path}")
            
            # Сохраняем отдельно уменьшенную версию
            comparison_small_path: str = os.path.join(test_dir, "comparisons", "methods_comparison_small.jpg")
            plt.savefig(comparison_small_path, dpi=100, bbox_inches='tight')
        
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        # Сохраняем статистику
        self._save_statistics(all_stats, test_dir)
        
        # Сохраняем результаты
        self._save_results_summary(results, test_dir)
        
        self.results[test_dir] = results
        print(f"✅ Тестирование завершено. Результаты в: {test_dir}")
        print(f"📋 Протестировано методов: {len(results)}/{len(method_names)}")
        
        return results
    
    def _save_statistics(self, 
                         stats: List[Dict], 
                         output_dir: str
    ) -> None:
        """Сохраняет статистику тестирования"""
        
        # Сохраняем как JSON
        stats_json_path: str = os.path.join(output_dir, "statistics", "statistics.json")
        with open(stats_json_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        # Сохраняем как CSV (если pandas доступен)
        try:
            df: pd.DataFrame = pd.DataFrame(stats)
            stats_csv_path: str = os.path.join(output_dir, "statistics", "statistics.csv")
            df.to_csv(stats_csv_path, index=False)
            print(f"📈 Статистика сохранена (CSV): {stats_csv_path}")
        except:
            pass
        
        # Создаем текстовый отчет
        report_path: str = os.path.join(output_dir, "statistics", "test_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("ОТЧЕТ О ТЕСТИРОВАНИИ МЕТОДОВ СЕГМЕНТАЦИИ\n")
            f.write("="*60 + "\n\n")
            f.write(f"Дата тестирования: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"ID теста: {self.current_test_id}\n")
            f.write(f"Всего методов: {len(stats)}\n\n")
            
            # Успешные методы
            successful: List[Dict] = [s for s in stats if 'error' not in s]
            if successful:
                f.write("УСПЕШНЫЕ МЕТОДЫ:\n")
                f.write("-"*40 + "\n")
                for stat in successful:
                    f.write(f"{stat['method']}:\n")
                    f.write(f"  Время: {stat['time_seconds']:.3f} сек\n")
                    f.write(f"  Площадь маски: {stat['mask_area_pixels']:,} пикселей\n")
                    f.write(f"  Процент покрытия: {stat['mask_percentage']:.2f}%\n")
                    if 'image_shape' in stat:
                        f.write(f"  Размер результата: {stat['image_shape']}\n")
                    f.write("\n")
            
            # Методы с ошибками
            failed: List[Dict] = [s for s in stats if 'error' in s]
            if failed:
                f.write("МЕТОДЫ С ОШИБКАМИ:\n")
                f.write("-"*40 + "\n")
                for stat in failed:
                    f.write(f"{stat['method']}: {stat['error']}\n")
        
        print(f"📋 Текстовый отчет сохранен: {report_path}")
    
    def _save_results_summary(self, 
                              results: Dict, 
                              output_dir: str
    ) -> None:
        """Сохраняет сводку результатов"""
        summary_path: str = os.path.join(output_dir, "statistics", "results_summary.json")
        
        # Подготовка данных для сохранения
        summary_data: Dict[str, Any] = {
            'test_id': self.current_test_id,
            'timestamp': datetime.now().isoformat(),
            'total_methods': len(results),
            'methods': {}
        }
        
        for method_name, result in results.items():
            method_data: Dict[str, Any] = {
                'time': result['time'],
                'mask_area': result['mask_area'],
                'mask_percentage': result['mask_percentage'],
                'image_shape': result['image_shape']
            }
            
            # Добавляем пути к файлам, если они есть
            for key in ['result_path', 'mask_path', 'overlay_path', 'original_path']:
                if key in result:
                    method_data[key] = result[key]
            
            summary_data['methods'][method_name] = method_data
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    def benchmark_methods(self, 
                          image: Union[str, np.ndarray, Image.Image], 
                          n_runs: int = 3,
                          save_benchmark: bool = True,
                          test_name: str = None,
                          save_results: bool = True
    ) -> pd.DataFrame:
        """Бенчмарк методов (требует pandas) с сохранением результатов"""
        
        benchmark_results = []

        # Создаем директорию для бенчмарка
        bench_dir: str
        if test_name:
            bench_dir = self._create_test_directory(f"benchmark_{test_name}")
        else:
            bench_dir = self._create_test_directory("benchmark")

        original_img: Image.Image
        image_array: np.ndarray
        orig_path: str
        if isinstance(image, str):
            original_img = Image.open(image).convert('RGB')
            image_array = np.array(original_img)
            # Сохраняем оригинальное изображение
            orig_path = os.path.join(bench_dir, "images", "original.jpg")
            original_img.save(orig_path)
            print(f"📸 Оригинальное изображение сохранено: {orig_path}")
        elif isinstance(image, Image.Image):
            original_img = image
            image_array = np.array(image)
        elif isinstance(image, np.ndarray):
            original_img = Image.fromarray(image.astype(np.uint8))
            image_array = image
            orig_path = os.path.join(bench_dir, "images", "original.jpg")
            original_img.save(orig_path)
        
        print(f"🏃 Запуск бенчмарка ({n_runs} прогонов)...")
        
        for method_name in self.methods.keys():
            print(f"  📊 Тестируем {method_name}...")
            
            times: List[float] = []
            results_list: List[np.ndarray] = []
            masks_list: List[np.ndarray] = []

            for run in range(n_runs):
                start_time: float = time.time()
                result: np.ndarray
                mask: np.ndarray
                result, mask = self.methods[method_name].segment_with_mask(image)
                times.append(time.time() - start_time)
                if run == 0:  # Сохраняем только первую маску для статистики
                    masks_list.append(mask)
                    results_list.append(result)
            
            mask_area: np.bool
            total_pixels: int
            if masks_list and results_list:
                mask = masks_list[0]
                result_img: np.ndarray = results_list[0]
                mask_area = np.sum(mask > 0)
                total_pixels = mask.shape[0] * mask.shape[1]
            else:
                mask_area = 0
                total_pixels = 1
            
            mean_time = np.mean(times)
            std_time = np.std(times)

            if save_results and masks_list and results_list:
                try:
                    # Сохраняем результат сегментации
                    result_path: str = os.path.join(bench_dir, "images", f"{method_name}_result.jpg")
                    result_pil: Image.Image = Image.fromarray(result_img.astype(np.uint8))
                    result_pil.save(result_path)
                    
                    # Сохраняем маску
                    mask_path: str = os.path.join(bench_dir, "masks", f"{method_name}_mask.png")
                    mask_pil: Image.Image = Image.fromarray(mask.astype(np.uint8))
                    mask_pil.save(mask_path)
                    
                    # Сохраняем overlay (50% оригинал + 90% результат)
                    if image_array is not None:
                        overlay: np.ndarray = image_array * 0.5 + result_img * 0.9
                        overlay: np.ndarray = overlay.astype(np.uint8)
                        overlay_path: str = os.path.join(bench_dir, "images", f"{method_name}_overlay.jpg")
                        overlay_pil: Image.Image = Image.fromarray(overlay)
                        overlay_pil.save(overlay_path)
                        
                        print(f"    💾 Результаты сохранены в {bench_dir}")
                except Exception as e:
                    print(f"    ⚠️ Ошибка сохранения результатов: {e}")
            
            benchmark_results.append({
                'Method': method_name,
                'Mean_Time_s': mean_time,
                'Std_Time_s': std_time,
                'Time_String': f"{mean_time:.3f} ± {std_time:.3f}",
                'Mask_Area': mask_area,
                'Mask_Percentage': (mask_area / total_pixels * 100) if total_pixels > 0 else 0,
                'Min_Time_s': min(times) if times else 0,
                'Max_Time_s': max(times) if times else 0,
                'Num_Runs': n_runs
            })
        
        df: pd.DataFrame = pd.DataFrame(benchmark_results)
        if 'Mean_Time_s' in df.columns:
            df = df.sort_values('Mean_Time_s')
        elif 'Mean_Time_s (s)' in df.columns:
            df = df.sort_values('Mean_Time_s (s)')
            
        print("\n" + "="*80)
        print("РЕЗУЛЬТАТЫ БЕНЧМАРКА:")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80)
        
        # Сохраняем результаты бенчмарка
        if save_benchmark:
            self._save_benchmark_results(df, bench_dir)
        
        return df
    
    def _save_benchmark_results(self, 
                                df: pd.DataFrame, 
                                output_dir: str
    ) -> None:
        """Сохраняет результаты бенчмарка"""
        bench_stats_dir: str = os.path.join(output_dir, "statistics")
        os.makedirs(bench_stats_dir, exist_ok=True)
        
        # Сохраняем как CSV
        csv_path: str = os.path.join(bench_stats_dir, "benchmark_results.csv")
        df.to_csv(csv_path, index=False)
        
        # Сохраняем как Excel (если установлен openpyxl)
        try:
            excel_path: str = os.path.join(bench_stats_dir, "benchmark_results.xlsx")
            df.to_excel(excel_path, index=False)
            print(f"📊 Excel отчет сохранен: {excel_path}")
        except:
            pass
        
        # Создаем текстовый отчет
        report_path: str = os.path.join(bench_stats_dir, "benchmark_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("ОТЧЕТ О БЕНЧМАРКЕ МЕТОДОВ СЕГМЕНТАЦИИ\n")
            f.write("="*80 + "\n\n")
            f.write(f"Дата тестирования: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Количество прогонов: {df['Num_Runs'].iloc[0] if not df.empty else 0}\n")
            f.write(f"Всего методов: {len(df)}\n\n")
            
            f.write("ТАБЛИЦА РЕЗУЛЬТАТОВ:\n")
            f.write("-"*80 + "\n")
            
            # Заголовок таблицы
            f.write(f"{'Метод':<30} {'Время (с)':<20} {'Площадь маски':<15} {'Процент':<10}\n")
            f.write("-"*80 + "\n")
            
            # Данные таблицы
            for _, row in df.iterrows():
                time_str = row.get('Time_String', f"{row.get('Mean_Time_s', 0):.3f} ± {row.get('Std_Time_s', 0):.3f}")
                mask_area: int = int(row.get('Mask_Area', 0))
                mask_percentage: float = row.get('Mask_Percentage', 0)
                f.write(f"{row['Method']:<30} {time_str:<20} "
                    f"{mask_area:<15,} {mask_percentage:.1f}%\n")
            
            # Сводка
            f.write("\n" + "="*80 + "\n")
            f.write("СВОДКА:\n")
            f.write("-"*80 + "\n")
            if not df.empty:
                fastest = df.iloc[0]
                slowest = df.iloc[-1]
                f.write(f"Самый быстрый метод: {fastest['Method']} ({fastest['Mean_Time_s']:.3f} с)\n")
                f.write(f"Самый медленный метод: {slowest['Method']} ({slowest['Mean_Time_s']:.3f} с)\n")
                f.write(f"Среднее время: {df['Mean_Time_s'].mean():.3f} с\n")
                f.write(f"Стандартное отклонение: {df['Mean_Time_s'].std():.3f} с\n")
        
        print(f"📋 Отчет бенчмарка сохранен: {report_path}")
        print(f"📊 CSV с результатами: {csv_path}")
        
        # Визуализация результатов бенчмарка
        self._plot_benchmark_results(df, output_dir)
    
    def _plot_benchmark_results(self, 
                                df: pd.DataFrame, 
                                output_dir: str
    ) -> None:
        """Создает графики результатов бенчмарка"""
        
        if df.empty:
            return
        
        # Создаем папку для сравнений
        comp_dir: str = os.path.join(output_dir, "comparisons")
        os.makedirs(comp_dir, exist_ok=True)
        
        # График 1: Время выполнения
        plt.figure(figsize=(12, 6))
        bars: plt.BarContainer = plt.barh(df['Method'], df['Mean_Time_s'])
        plt.xlabel('Время выполнения (секунды)')
        plt.title('Бенчмарк методов сегментации: Время выполнения')
        
        # Добавляем ошибки (стандартное отклонение)
        if 'Std_Time_s' in df.columns:
            plt.errorbar(df['Mean_Time_s'], df['Method'], 
                        xerr=df['Std_Time_s'], 
                        fmt='none', ecolor='black', capsize=5)
        
        # Добавляем значения на столбцы
        max_time = df['Mean_Time_s'].max() if not df.empty else 0
        for bar, time_val in zip(bars, df['Mean_Time_s']):
            plt.text(time_val + max_time * 0.01, 
                    bar.get_y() + bar.get_height()/2,
                    f'{time_val:.3f}s', 
                    va='center', fontsize=9)
        
        plt.tight_layout()
        bench_plot_path: str = os.path.join(comp_dir, "benchmark_time.png")
        plt.savefig(bench_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # График 2: Время vs Площадь маски
        plt.figure(figsize=(10, 6))
        if 'Mask_Percentage' in df.columns:
            scatter: plt.PathCollection = plt.scatter(df['Mean_Time_s'], df['Mask_Percentage'], 
                                s=100, alpha=0.7, c=range(len(df)), cmap='viridis')
            
            # Подписи точек
            for i, (x, y, method) in enumerate(zip(df['Mean_Time_s'], df['Mask_Percentage'], df['Method'])):
                plt.annotate(method, (x, y), textcoords="offset points", 
                            xytext=(0,10), ha='center', fontsize=9)
            
            plt.xlabel('Время выполнения (секунды)')
            plt.ylabel('Площадь маски (%)')
            plt.title('Бенчмарк: Время vs Площадь покрытия')
            plt.colorbar(scatter, label='Ранг метода')
            plt.grid(True, alpha=0.3)
            
            bench_scatter_path: str = os.path.join(comp_dir, "benchmark_scatter.png")
            plt.savefig(bench_scatter_path, dpi=150, bbox_inches='tight')
            plt.close()

        # График 3: Сравнительная визуализация результатов (маленькие превью)
        self._create_benchmark_preview(df, output_dir, comp_dir)
        
        print(f"📈 Графики бенчмарка сохранены в {output_dir}/comparisons/")

    def _create_benchmark_preview(self, 
                                  df: pd.DataFrame, 
                                  output_dir: str, 
                                  comp_dir: str
    ) -> None:
        """Создает превью результатов всех методов"""
        
        # Получаем все изображения результатов
        images_dir: str = os.path.join(output_dir, "images")
        result_files: List[str] = [f for f in os.listdir(images_dir) if f.endswith('_result.jpg')]
        
        if not result_files:
            return
        
        # Сортируем файлы по времени выполнения (согласно бенчмарку)
        sorted_methods: List[Any] = df.sort_values('Mean_Time_s')['Method'].tolist()
        
        # Загружаем изображения
        images = []
        titles = []
        for method in sorted_methods:
            result_file: str = f"{method}_result.jpg"
            if result_file in result_files:
                img_path: str = os.path.join(images_dir, result_file)
                img: Image.Image = Image.open(img_path)
                images.append(img)
                
                # Формируем заголовок
                method_data = df[df['Method'] == method]
                if not method_data.empty:
                    time_val = method_data.iloc[0]['Mean_Time_s']
                    mask_percent: float = method_data.iloc[0]['Mask_Percentage'] if 'Mask_Percentage' in method_data.columns else 0
                    title: str = f"{method}\n{time_val:.3f}s, {mask_percent:.1f}%"
                else:
                    title = method
                titles.append(title)
        
        # Создаем сетку превью
        n_images: int = len(images)
        n_cols: int = 4
        n_rows: int = (n_images + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3))
        axes = axes.flatten()
        
        for i, (img, title) in enumerate(zip(images, titles)):
            axes[i].imshow(img)
            axes[i].set_title(title, fontsize=8)
            axes[i].axis('off')
        
        # Скрываем пустые оси
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        
        plt.suptitle("Превью результатов сегментации (сортировка по скорости)", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        preview_path:str = os.path.join(comp_dir, "methods_preview.jpg")
        plt.savefig(preview_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def visualize_comparison(self, 
                             results: Dict[str, Dict], 
                             show_masks: bool = True,
                             save_visualization: bool = True,
                             output_dir: str = None,
                             show_plots: bool = True
    ) -> None:
        """Визуализация сравнения результатов с сохранением"""
        output_dir: str
        if output_dir is None and self.current_test_id:
            output_dir = os.path.join(self.base_output_dir, self.current_test_id)
        elif output_dir is None:
            output_dir = self._create_test_directory("visualization")
        n_methods: int = len(results)
        
        if show_masks:
            fig, axes = plt.subplots(2, n_methods, figsize=(5 * n_methods, 10))
            
            for i, (method_name, result) in enumerate(results.items()):
                # Результат
                axes[0, i].imshow(result['result'])
                title: str = f"{method_name}\n{result['time']:.2f}s"
                axes[0, i].set_title(title, fontsize=10)
                axes[0, i].axis('off')
                
                # Маска
                axes[1, i].imshow(result['mask'], cmap='gray')
                mask_title: str = f"Mask\n{result['mask_percentage']:.1f}%"
                axes[1, i].set_title(mask_title, fontsize=10)
                axes[1, i].axis('off')
        else:
            fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 5))
            
            for i, (method_name, result) in enumerate(results.items()):
                axes[i].imshow(result['result'])
                title: str = f"{method_name}\n{result['time']:.2f}s, {result['mask_percentage']:.1f}%"
                axes[i].set_title(title, fontsize=10)
                axes[i].axis('off')
        
        plt.suptitle("Визуализация результатов сегментации", fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Сохраняем визуализацию
        if save_visualization:
            vis_dir: str = os.path.join(output_dir, "comparisons")
            os.makedirs(vis_dir, exist_ok=True)
            
            vis_path: str = os.path.join(vis_dir, "results_visualization.jpg")
            plt.savefig(vis_path, dpi=150, bbox_inches='tight')
            print(f"🖼️ Визуализация сохранена: {vis_path}")
            
            # Сохраняем отдельно для каждого метода
            for method_name, result in results.items():
                # Сохраняем увеличенный результат
                result_fig, result_ax = plt.subplots(1, 1, figsize=(8, 6))
                result_ax.imshow(result['result'])
                result_ax.set_title(f"{method_name} - Result", fontsize=12)
                result_ax.axis('off')
                
                result_path: str = os.path.join(output_dir, "images", f"{method_name}_large.jpg")
                result_fig.savefig(result_path, dpi=150, bbox_inches='tight')
                plt.close(result_fig)
                
                # Сохраняем увеличенную маску
                mask_fig, mask_ax = plt.subplots(1, 1, figsize=(8, 6))
                mask_ax.imshow(result['mask'], cmap='gray')
                mask_ax.set_title(f"{method_name} - Mask", fontsize=12)
                mask_ax.axis('off')
                
                mask_path: str = os.path.join(output_dir, "masks", f"{method_name}_large_mask.jpg")
                mask_fig.savefig(mask_path, dpi=150, bbox_inches='tight')
                plt.close(mask_fig)
        
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    def save_results(self, 
                     results: Dict[str, Dict], 
                     output_dir: str = "segmentation_results"
    ) -> None:
        """Сохранение результатов всех методов"""
    
        if output_dir is None:
            output_dir = self._create_test_directory("results_save")
        
        print(f"💾 Сохранение результатов в {output_dir}...")
        
        for method_name, result in results.items():
            # Сохраняем результат
            result_path: str = os.path.join(output_dir, f"{method_name}_result.jpg")
            result_img: Image.Image = Image.fromarray(result['result'].astype(np.uint8))
            result_img.save(result_path)
            
            # Сохраняем маску
            mask_path: str = os.path.join(output_dir, f"{method_name}_mask.png")
            mask_img: Image.Image = Image.fromarray(result['mask'].astype(np.uint8))
            mask_img.save(mask_path)
            
            # Сохраняем статистику
            stats_path: str = os.path.join(output_dir, f"{method_name}_stats.txt")
            with open(stats_path, 'w') as f:
                f.write(f"Method: {method_name}\n")
                f.write(f"Execution Time: {result['time']:.3f}s\n")
                f.write(f"Mask Area: {result['mask_area']} pixels\n")
                f.write(f"Mask Percentage: {result['mask_percentage']:.2f}%\n")
                f.write(f"Image Shape: {result['image_shape']}\n")
                if 'timestamp' in result:
                    f.write(f"Timestamp: {result['timestamp']}\n")
        
        print(f"✅ Все результаты сохранены в директории: {output_dir}")