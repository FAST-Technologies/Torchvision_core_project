# testing/BatchClassicTester.py
import os
import signal
import sys
import time
import json
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from segmenters.OpenCVSegmenter import OpenCVSegmenter
from segmenters.SklearnSegmenter import SklearnSegmenter
from segmenters.TorchSegmenter import TorchSegmenter
from metrics.SegmentationMetrics import SegmentationMetrics
import torch
import gc


class BatchClassicTester:
    """
    Массовое тестирование классических методов сегментации на множестве изображений.
    Рассчитывает среднюю погрешность и другие агрегированные метрики.
    Автосохранение прогресса.
    Прогресс-бар с ETA.
    Восстановление после сбоя.
    """

    def __init__(
        self,
        ade20k_root: str = "./data/ade20k/ADEChallengeData2016",
        output_dir: str = "./data/batch_classic_test",
        split: str = "validation",  # "training" или "validation"
        max_images: Optional[int] = None,  # Лимит изображений для теста
        image_size: Tuple[int, int] = (512, 512),
        autosave_interval: int = 5,
        resume: bool = True,
    ) -> None:
        _setup_signal_handlers()
        self.ade20k_root = Path(ade20k_root)
        self.output_dir = Path(output_dir)
        self.split = split
        self.max_images = max_images
        self.image_size = image_size

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Метрики для агрегации
        self.metrics_to_aggregate = [
            "iou",
            "dice",
            "precision",
            "recall",
            "f1_score",
            "accuracy",
            "mae",
            "hausdorff_distance",
        ]

        # Результаты
        self.results: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.execution_times: Dict[str, List[float]] = defaultdict(list)
        self.errors: Dict[str, List[str]] = defaultdict(list)
        self.autosave_interval = autosave_interval
        self.resume = resume

        # Пути для автосохранения
        self.progress_file = Path(output_dir) / ".progress.json"
        self.temp_results_file = Path(output_dir) / ".results_temp.csv"

        # Загрузка предыдущего прогресса если нужно
        if resume and self.progress_file.exists():
            self._load_progress()
            print(
                f"📥 Восстановлен прогресс: {self._processed_count}/{self._total_tests} тестов"
            )

    # ──────────────────────────────────────────────────────────────────────
    def _setup_signal_handlers(self):
        """Настройка обработчиков сигналов для автосохранения при прерывании"""

        def handle_interrupt(signum, frame):
            print("\n\n⚠️  Получен сигнал прерывания!")
            print("💾 Выполняется финальное автосохранение...")
            self._save_progress(force=True)
            print("✅ Прогресс сохранён. Завершение работы.")
            sys.exit(130)  # Стандартный код для SIGINT

        signal.signal(signal.SIGINT, handle_interrupt)  # Ctrl+C
        signal.signal(signal.SIGTERM, handle_interrupt)  # kill

    # ──────────────────────────────────────────────────────────────────────
    def _init_progress_tracking(self, total_images: int, total_methods: int):
        """Инициализация трекера прогресса"""
        self._total_images = total_images
        self._total_methods = total_methods
        self._total_tests = total_images * total_methods
        self._processed_count = getattr(self, "_processed_count", 0)
        self._start_time = time.time()
        self._last_save_time = time.time()

    # ──────────────────────────────────────────────────────────────────────
    def _update_progress_bar(
        self, pbar, current_count: int, method_name: str, img_name: str
    ):
        """Обновление прогресс-бара с оценкой времени"""
        elapsed = time.time() - self._start_time
        rate = current_count / elapsed if elapsed > 0 else 0
        remaining = (self._total_tests - current_count) / rate if rate > 0 else 0

        # Форматирование времени
        def fmt_time(seconds):
            if seconds < 60:
                return f"{seconds:.0f}с"
            elif seconds < 3600:
                return f"{seconds/60:.1f}м"
            else:
                return f"{seconds/3600:.1f}ч"

        total_errors = sum(len(errs) for errs in self.errors.values())
        error_rate = total_errors / current_count if current_count > 0 else 0

        pbar.set_postfix(
            {
                "img": img_name[:15],
                "method": method_name.split("_")[0],
                "elapsed": fmt_time(elapsed),
                "eta": fmt_time(remaining),
                "rate": f"{rate*60:.1f}/мин",
                "errors": f"{total_errors}({error_rate*100:.1f}%)",
            }
        )

    # ──────────────────────────────────────────────────────────────────────
    def _save_progress(self, force: bool = False):
        """Автосохранение прогресса и результатов"""
        now = time.time()
        # Сохраняем если прошло достаточно времени или по принуждению
        if not force and (now - self._last_save_time) < 30:  # не чаще 30 сек
            return

        try:
            # 1. Сохраняем метаданные прогресса
            progress = {
                "processed_count": self._processed_count,
                "total_tests": self._total_tests,
                "start_time": self._start_time,
                "last_update": now,
                "methods_done": list(self.results.keys()),
            }
            with open(self.progress_file, "w") as f:
                json.dump(progress, f, indent=2)

            # 2. Сохраняем промежуточные результаты (atomic write)
            if self.results:
                df_temp = self._aggregate_results()
                df_temp.to_csv(self.temp_results_file, index=False, float_format="%.4f")
                # Atomic rename
                final_path = self.temp_results_file.with_suffix(".csv")
                self.temp_results_file.replace(final_path)

            self._last_save_time = now
            print(
                f"\n💾 Автосохранение: {self._processed_count}/{self._total_tests} ✅"
            )

        except Exception as e:
            print(f"\n⚠️  Ошибка автосохранения: {e}")

    # ──────────────────────────────────────────────────────────────────────
    def _load_progress(self) -> bool:
        """Загрузка предыдущего прогресса"""
        try:
            with open(self.progress_file, "r") as f:
                progress = json.load(f)

            # Восстанавливаем счётчики
            self._processed_count = progress.get("processed_count", 0)
            self._total_tests = progress.get("total_tests", 0)
            self._start_time = progress.get("start_time", time.time())

            # Если есть временные результаты — загружаем их
            if self.temp_results_file.exists():
                print(
                    f"📥 Загрузка промежуточных результатов из {self.temp_results_file}"
                )
                # Здесь можно добавить логику слияния, если нужно

            return True
        except Exception as e:
            print(f"⚠️  Не удалось загрузить прогресс: {e}")
            return False

    def _load_images_with_masks(self) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        """
        Загружает изображения и их ground truth маски из ADE20K.

        Returns:
            List[Tuple[name, image_array, mask_array]]
        """
        images_dir = self.ade20k_root / "images" / self.split
        masks_dir = self.ade20k_root / "annotations" / self.split

        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        if not masks_dir.exists():
            raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

        # Получаем список изображений
        image_files = sorted(
            [f for f in os.listdir(images_dir) if f.endswith((".jpg", ".jpeg", ".png"))]
        )

        if self.max_images:
            image_files = image_files[: self.max_images]

        print(f"📥 Загрузка {len(image_files)} изображений из {self.split}...")

        data = []
        for img_file in tqdm(image_files, desc="Loading images"):
            mask_file = img_file.rsplit(".", 1)[0] + ".png"
            img_path = images_dir / img_file
            mask_path = masks_dir / mask_file

            if not mask_path.exists():
                print(f"⚠️  Mask not found for {img_file}, skipping")
                continue

            try:
                # Загрузка изображения
                img = Image.open(img_path).convert("RGB")
                img_array = np.array(
                    img.resize(self.image_size, Image.Resampling.BILINEAR)
                )

                # Загрузка маски
                mask_pil = Image.open(mask_path)
                mask_array = np.array(
                    mask_pil.resize(self.image_size, Image.Resampling.NEAREST)
                )

                # Конвертация многоклассовой маски в бинарную
                # Стратегия: самый частый класс = фон (0), всё остальное = объект (255)
                binary_mask = self._multiclass_to_binary(mask_array)

                data.append((img_file, img_array, binary_mask))

            except Exception as e:
                print(f"❌ Error loading {img_file}: {e}")
                continue

        print(f"✅ Загружено {len(data)} пар изображение-маска")
        return data

    def _multiclass_to_binary(self, mask: np.ndarray) -> np.ndarray:
        """
        Конвертирует многоклассовую маску ADE20K в бинарную.

        Стратегия: самый частый класс считается фоном, остальные — объектом.
        """
        # Находим самый частый класс (предполагаем, что это фон)
        unique, counts = np.unique(mask, return_counts=True)
        background_class = unique[np.argmax(counts)]

        # Бинаризация: объект = всё, что не фон
        binary = (mask != background_class).astype(np.uint8) * 255
        return binary

    def _get_classic_methods(self) -> Dict[str, Any]:
        """Возвращает словарь классических методов для тестирования."""
        methods = {}

        # === OpenCV методы ===
        methods.update(
            {
                "Global_Threshold_CV2": OpenCVSegmenter(
                    "global_thresholding", threshold=0.5
                ),
                "Otsu_Thresholding_CV2": OpenCVSegmenter("otsu_thresholding"),
                "Adaptive_Threshold_CV2": OpenCVSegmenter(
                    "adaptive_thresholding", block_size=11, C=2
                ),
                "Niblack_Thresholding_CV2": OpenCVSegmenter(
                    "threshold_niblack", window_size=15, k=-0.2
                ),
                "Sauvola_Thresholding_CV2": OpenCVSegmenter(
                    "threshold_sauvola", window_size=15, k=0.5, r=128
                ),
                "Sobel_CV2": OpenCVSegmenter("sobel_edge", threshold=0.1),
                "Canny_CV2": OpenCVSegmenter(
                    "canny_edge", low=0.1, high=0.3, sigma=1.0
                ),
            }
        )

        # === Sklearn методы ===
        methods.update(
            {
                "Global_Threshold_Sklearn": SklearnSegmenter(
                    "global_thresholding", threshold=0.5
                ),
                "Otsu_Thresholding_Sklearn": SklearnSegmenter("otsu_thresholding"),
                "Adaptive_Threshold_Sklearn": SklearnSegmenter(
                    "adaptive_thresholding", block_size=11, C=2
                ),
                "Niblack_Thresholding_Sklearn": SklearnSegmenter(
                    "threshold_niblack", window_size=15, k=-0.2
                ),
                "Sauvola_Thresholding_Sklearn": SklearnSegmenter(
                    "threshold_sauvola", window_size=15, k=0.5, r=128
                ),
                "Sobel_Sklearn": SklearnSegmenter("sobel_edge", threshold=0.1),
                "Canny_Sklearn": SklearnSegmenter(
                    "canny_edge", low=0.1, high=0.3, sigma=1.0
                ),
            }
        )

        # === Torch методы ===
        methods.update(
            {
                "Global_Threshold_Torch": TorchSegmenter(
                    "global_thresholding", threshold=0.5
                ),
                "Otsu_Thresholding_Torch": TorchSegmenter("otsu_thresholding"),
                "Adaptive_Threshold_Torch": TorchSegmenter(
                    "adaptive_thresholding", block_size=11, C=2
                ),
                "Niblack_Thresholding_Torch": TorchSegmenter(
                    "threshold_niblack", window_size=15, k=-0.2
                ),
                "Sauvola_Thresholding_Torch": TorchSegmenter(
                    "threshold_sauvola", window_size=15, k=0.5, r=128
                ),
                "Sobel_Torch": TorchSegmenter("sobel_edge", threshold=0.1),
                "Canny_Torch": TorchSegmenter(
                    "canny_edge", low=0.1, high=0.3, sigma=1.0
                ),
            }
        )

        return methods

    def _run_single_test(
        self, method_name: str, segmenter: Any, image: np.ndarray, gt_mask: np.ndarray
    ) -> Tuple[Optional[Dict[str, float]], Optional[float], Optional[str]]:
        """
        Запускает один тест для метода на одном изображении.

        Returns:
            (metrics_dict, execution_time, error_message)
        """
        try:
            start_time = time.time()

            # Сегментация
            pred_mask = segmenter.segment(image)

            # Ресайз предсказания к размеру GT если нужно
            if pred_mask.shape != gt_mask.shape:
                from skimage.transform import resize

                pred_mask = resize(
                    pred_mask, gt_mask.shape, order=0, preserve_range=True
                ).astype(np.uint8)

            exec_time = time.time() - start_time

            # Расчёт метрик
            metrics = SegmentationMetrics.calculate_all_metrics(
                pred_mask=pred_mask,
                gt_mask=gt_mask,
                threshold=0.5,
                include_hausdorff=True,
            )

            return metrics, exec_time, None

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            return None, None, error_msg

    def run_batch_test(self) -> pd.DataFrame:
        """
        Запускает массовое тестирование всех методов на всех изображениях.

        Returns:
            DataFrame с агрегированными результатами
        """
        # Загрузка данных
        test_data = self._load_images_with_masks()
        if not test_data:
            raise ValueError("No test data loaded!")

        # Получение методов
        methods = self._get_classic_methods()
        total_images = len(test_data)
        total_methods = len(methods)

        print(f"🔧 Тестируем {total_methods} методов на {total_images} изображениях")
        print(f"📊 Всего тестов: {total_images * total_methods}")
        print(f"💾 Автосохранение каждые {self.autosave_interval} изображений")

        # Инициализация прогресса
        self._init_progress_tracking(total_images, total_methods)

        # Прогресс-бар с расширенными настройками
        from tqdm import tqdm

        with tqdm(
            total=self._total_tests,
            desc="🧪 Тестирование",
            unit="тест",
            leave=True,
            dynamic_ncols=True,
            mininterval=0.5,
        ) as pbar:

            # Пропускаем уже выполненные тесты если resume=True
            if self.resume and self._processed_count > 0:
                pbar.update(self._processed_count)
                print(f"⏭️  Пропущено {self._processed_count} выполненных тестов")

            # Основной цикл
            for img_idx, (img_name, image, gt_mask) in enumerate(test_data):
                for method_idx, (method_name, segmenter) in enumerate(methods.items()):

                    # Пропуск если уже выполнено (при resume)
                    test_key = f"{img_name}:{method_name}"
                    if self.resume and self._processed_count > 0:
                        # Простая эвристика: если счётчик больше — пропускаем
                        if self._processed_count >= (
                            img_idx * total_methods + method_idx + 1
                        ):
                            continue

                    # Запуск теста
                    metrics, exec_time, error = self._run_single_test(
                        method_name, segmenter, image, gt_mask
                    )

                    # Обработка результатов
                    if error:
                        self.errors[method_name].append(f"{img_name}: {error}")
                    elif metrics:
                        for metric_name in self.metrics_to_aggregate:
                            if metric_name in metrics:
                                self.results[method_name][metric_name].append(
                                    metrics[metric_name]
                                )
                        if exec_time is not None:
                            self.execution_times[method_name].append(exec_time)
                    # Обновление прогресса
                    self._processed_count += 1
                    pbar.update(1)
                    self._update_progress_bar(
                        pbar, self._processed_count, method_name, img_name
                    )

                    # Автосохранение каждые N изображений
                    if img_idx % self.autosave_interval == 0 and img_idx > 0:
                        self._save_progress()

                # Очистка памяти после каждого изображения
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                gc.collect()

        # Финальное сохранение
        print("\n🏁 Завершение тестирования...")
        self._save_progress(force=True)

        # Убираем временные файлы
        if self.progress_file.exists():
            self.progress_file.unlink()
        if self.temp_results_file.exists():
            self.temp_results_file.unlink()

        return self._aggregate_results()

    def _aggregate_results(self) -> pd.DataFrame:
        """Агрегирует результаты и создаёт сводный DataFrame."""
        rows = []

        for method_name in self.results:
            row = {
                "Method": method_name,
                "Images_Tested": len(self.results[method_name].get("iou", [])),
            }

            # Средние значения метрик
            for metric_name in self.metrics_to_aggregate:
                values = self.results[method_name].get(metric_name, [])
                if values:
                    row[f"{metric_name}_mean"] = np.mean(values)
                    row[f"{metric_name}_std"] = np.std(values)
                    row[f"{metric_name}_min"] = np.min(values)
                    row[f"{metric_name}_max"] = np.max(values)
                else:
                    row[f"{metric_name}_mean"] = np.nan

            # Время выполнения
            times = self.execution_times.get(method_name, [])
            if times:
                row["time_mean_s"] = np.mean(times)
                row["time_std_s"] = np.std(times)
            else:
                row["time_mean_s"] = np.nan

            # Ошибки
            errors = self.errors.get(method_name, [])
            row["error_count"] = len(errors)
            row["error_rate"] = len(errors) / max(row["Images_Tested"], 1)

            rows.append(row)

        df = pd.DataFrame(rows)

        # Сортировка по IoU
        if "iou_mean" in df.columns:
            df = df.sort_values("iou_mean", ascending=False)

        return df

    def save_results(self, df: pd.DataFrame, prefix: str = "batch_test"):
        """Сохраняет результаты в различные форматы."""
        # CSV
        csv_path = self.output_dir / f"{prefix}_results.csv"
        df.to_csv(csv_path, index=False, float_format="%.4f")
        print(f"💾 CSV сохранён: {csv_path}")

        # JSON с детальными результатами
        json_path = self.output_dir / f"{prefix}_details.json"
        details = {
            "summary": df.to_dict(orient="records"),
            "errors": dict(self.errors),
            "config": {
                "ade20k_root": str(self.ade20k_root),
                "split": self.split,
                "max_images": self.max_images,
                "image_size": self.image_size,
            },
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(details, f, indent=2, ensure_ascii=False, default=str)
        print(f"💾 JSON сохранён: {json_path}")

        # Markdown отчёт
        md_path = self.output_dir / f"{prefix}_report.md"
        self._save_markdown_report(df, md_path)
        print(f"💾 Markdown отчёт: {md_path}")

        return csv_path, json_path, md_path

    def _save_markdown_report(self, df: pd.DataFrame, path: Path):
        """Генерирует Markdown-отчёт."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(
                "# 📊 Отчёт: Массовое тестирование классических методов сегментации\n\n"
            )
            f.write(f"**Датасет:** ADE20K ({self.split})\n")
            f.write(f"**Изображений:** {self.max_images or 'all'}\n")
            f.write(f"**Размер:** {self.image_size}\n\n")

            # Топ-10 по IoU
            f.write("## 🏆 Топ-10 методов по среднему IoU\n\n")
            top_10 = df.head(10)[
                ["Method", "iou_mean", "dice_mean", "time_mean_s", "Images_Tested"]
            ]
            f.write(top_10.to_markdown(index=False) + "\n\n")

            # Сводная таблица всех метрик
            f.write("## 📈 Полная таблица результатов\n\n")
            cols_to_show = ["Method"] + [c for c in df.columns if c.endswith("_mean")]
            f.write(df[cols_to_show].to_markdown(index=False, floatfmt=".4f") + "\n\n")

            # Статистика ошибок
            if any(df["error_count"] > 0):
                f.write("## ⚠️ Статистика ошибок\n\n")
                error_df = df[df["error_count"] > 0][
                    ["Method", "error_count", "error_rate"]
                ]
                f.write(error_df.to_markdown(index=False) + "\n\n")

    def plot_results(self, df: pd.DataFrame, output_dir: Optional[Path] = None):
        """Строит графики результатов."""
        if output_dir is None:
            output_dir = self.output_dir / "charts"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Фильтруем методы с достаточным количеством тестов
        df_plot = df[df["Images_Tested"] >= 5].copy()
        if df_plot.empty:
            print("⚠️ Недостаточно данных для построения графиков")
            return

        # === График 1: IoU по методам ===
        plt.figure(figsize=(14, 8))
        sns.barplot(data=df_plot.head(15), x="iou_mean", y="Method", palette="viridis")
        plt.xlabel("Mean IoU")
        plt.title("Топ-15 методов по среднему IoU")
        plt.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "iou_ranking.png", dpi=150)
        plt.close()

        # === График 2: Speed vs Accuracy ===
        plt.figure(figsize=(10, 8))
        plt.scatter(
            df_plot["time_mean_s"],
            df_plot["iou_mean"],
            s=100,
            alpha=0.7,
            edgecolors="black",
        )

        for _, row in df_plot.iterrows():
            plt.annotate(
                row["Method"][:15],
                (row["time_mean_s"], row["iou_mean"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=7,
            )

        plt.xlabel("Среднее время (сек)")
        plt.ylabel("Mean IoU")
        plt.title("Зависимость точности от скорости")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "speed_vs_accuracy.png", dpi=150)
        plt.close()

        # === График 3: Heatmap метрик ===
        metrics_cols = [
            c
            for c in df.columns
            if c.endswith("_mean") and c.split("_")[0] in self.metrics_to_aggregate
        ]
        if metrics_cols:
            plt.figure(figsize=(12, 10))
            heatmap_data = (
                df_plot[["Method"] + metrics_cols].set_index("Method").head(12)
            )
            heatmap_data.columns = [
                c.replace("_mean", "") for c in heatmap_data.columns
            ]

            sns.heatmap(heatmap_data.T, annot=True, fmt=".3f", cmap="YlOrRd")
            plt.title("Heatmap средних метрик (Топ-12 методов)")
            plt.xlabel("Метод")
            plt.ylabel("Метрика")
            plt.tight_layout()
            plt.savefig(output_dir / "metrics_heatmap.png", dpi=150)
            plt.close()

        print(f"📊 Графики сохранены в: {output_dir}")
