# testing/CpuCudaBenchmark.py

# Импорт основных библиотек
import os
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import (
    Dict,
    Any,

)
from datetime import datetime

class CpuCudaBenchmark:
    """
    Бенчмарк для сравнения производительности методов сегментации на CPU и CUDA.
    """

    def __init__(
        self,
        base_output_dir: str = "./data/cpu_cuda_benchmark",
        n_runs: int = 5,
        warmup_runs: int = 2,
    ):
        self.base_output_dir = base_output_dir
        self.n_runs = n_runs
        self.warmup_runs = warmup_runs

    def benchmark_method(
        self, segmenter: Any, image: np.ndarray, method_name: str, device: str = "cpu"
    ) -> Dict[str, Any]:
        """
        Тестирование одного метода на указанном устройстве.

        Args:
            segmenter: Экземпляр сегментера
            image: Изображение (numpy array)
            method_name: Имя метода
            device: "cpu" или "cuda"

        Returns:
            Dict со статистикой времени
        """
        times = []

        # Переключение устройства для Torch-сегментеров
        if hasattr(segmenter, "device"):
            original_device = segmenter.device
            segmenter.device = torch.device(device)
            if hasattr(segmenter, "model") and segmenter.model is not None:
                segmenter.model = segmenter.model.to(device)

        # Warm-up
        for _ in range(self.warmup_runs):
            try:
                segmenter.segment(image)
            except:
                pass

        # Основное тестирование
        torch.cuda.synchronize() if device == "cuda" else None
        start_total = time.perf_counter()

        for run in range(self.n_runs):
            try:
                if device == "cuda":
                    torch.cuda.synchronize()

                start = time.perf_counter()
                segmenter.segment(image)
                end = time.perf_counter()

                if device == "cuda":
                    torch.cuda.synchronize()

                times.append(end - start)

            except Exception as e:
                print(f"⚠️ Error in {method_name} ({device}), run {run+1}: {e}")
                break

        torch.cuda.synchronize() if device == "cuda" else None
        end_total = time.perf_counter()

        # Восстановление оригинального устройства
        if hasattr(segmenter, "device"):
            segmenter.device = original_device
            if hasattr(segmenter, "model") and segmenter.model is not None:
                segmenter.model = segmenter.model.to(original_device)

        if not times:
            return {
                "method": method_name,
                "device": device,
                "error": "Failed to execute",
                "mean_time": float("inf"),
                "std_time": 0,
                "min_time": float("inf"),
                "max_time": float("inf"),
                "total_time": end_total - start_total,
                "n_runs": 0,
            }

        return {
            "method": method_name,
            "device": device,
            "mean_time": np.mean(times),
            "std_time": np.std(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
            "total_time": end_total - start_total,
            "n_runs": len(times),
        }

    def benchmark_all_methods(
        self,
        methods_dict: Dict[str, Any],
        image: np.ndarray,
        test_name: str = "cpu_cuda_comparison",
    ) -> pd.DataFrame:
        """
        Тестирование всех методов на CPU и CUDA.

        Args:
            methods_dict: {method_name: segmenter_object}
            image: Тестовое изображение
            test_name: Имя теста для сохранения

        Returns:
            DataFrame с результатами
        """
        all_results = []

        print("\n" + "=" * 80)
        print("БЕНЧМАРК: CPU vs CUDA")
        print("=" * 80)
        print(f"Количество методов: {len(methods_dict)}")
        print(f"Количество прогонов: {self.n_runs}")
        print(f"Warm-up прогонов: {self.warmup_runs}")
        print("=" * 80)

        for method_name, segmenter in methods_dict.items():
            print(f"\n🔹 Метод: {method_name}")

            # Тест на CPU
            print(f"   📊 Тестирование на CPU...")
            cpu_result = self.benchmark_method(segmenter, image, method_name, "cpu")
            all_results.append(cpu_result)
            print(
                f"      CPU: {cpu_result['mean_time']*1000:.2f}ms ± {cpu_result['std_time']*1000:.2f}ms"
            )

            # Тест на CUDA (только если доступно)
            if torch.cuda.is_available():
                print("   📊 Тестирование на CUDA...")
                cuda_result = self.benchmark_method(
                    segmenter, image, method_name, "cuda"
                )
                all_results.append(cuda_result)
                print(
                    f"      CUDA: {cuda_result['mean_time']*1000:.2f}ms ± {cuda_result['std_time']*1000:.2f}ms"
                )

                # Ускорение
                if cpu_result["mean_time"] > 0 and cuda_result["mean_time"] > 0:
                    speedup = cpu_result["mean_time"] / cuda_result["mean_time"]
                    print(f"      ⚡ Ускорение: {speedup:.2f}x")
            else:
                print("   ⚠️ CUDA недоступна, пропускаем")

        # Создаем DataFrame
        df = pd.DataFrame(all_results)

        # Сохраняем результаты
        self._save_results(df, test_name)

        # Визуализация
        self._plot_results(df, test_name)

        return df

    def _save_results(self, df: pd.DataFrame, test_name: str):
        """Сохранение результатов"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(self.base_output_dir, f"{test_name}_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)

        # CSV
        csv_path = os.path.join(output_dir, "results.csv")
        df.to_csv(csv_path, index=False)

        # Excel с форматированием
        try:
            excel_path = os.path.join(output_dir, "results.xlsx")
            with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name="Results", index=False)

                # Авто-ширина колонок
                worksheet = writer.sheets["Results"]
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[column_letter].width = adjusted_width
        except Exception as e:
            print(f"⚠️ Не удалось сохранить Excel: {e}")

        # Текстовый отчёт
        report_path = os.path.join(output_dir, "report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("ОТЧЁТ: СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ CPU vs CUDA\n")
            f.write("=" * 80 + "\n")
            f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Количество методов: {df['method'].nunique()}\n")
            f.write(f"Количество прогонов: {self.n_runs}\n")
            f.write("=" * 80 + "\n\n")

            # Сводка по методам
            f.write("СВОДКА ПО МЕТОДАМ:\n")
            f.write("-" * 80 + "\n")

            for method in df["method"].unique():
                method_data = df[df["method"] == method]
                f.write(f"\n{method}:\n")

                cpu_data = method_data[method_data["device"] == "cpu"]
                if not cpu_data.empty:
                    f.write(
                        f"  CPU:  {cpu_data['mean_time'].values[0]*1000:.2f}ms ± {cpu_data['std_time'].values[0]*1000:.2f}ms\n"
                    )

                cuda_data = method_data[method_data["device"] == "cuda"]
                if not cuda_data.empty:
                    f.write(
                        f"  CUDA: {cuda_data['mean_time'].values[0]*1000:.2f}ms ± {cuda_data['std_time'].values[0]*1000:.2f}ms\n"
                    )

                    if not cpu_data.empty and cuda_data["mean_time"].values[0] > 0:
                        speedup = (
                            cpu_data["mean_time"].values[0]
                            / cuda_data["mean_time"].values[0]
                        )
                        f.write(f"  ⚡ Ускорение: {speedup:.2f}x\n")

            # Топ ускорений
            f.write("\n" + "=" * 80 + "\n")
            f.write("ТОП-5 МЕТОДОВ ПО УСКОРЕНИЮ (CUDA vs CPU):\n")
            f.write("=" * 80 + "\n")

            speedups = []
            for method in df["method"].unique():
                method_data = df[df["method"] == method]
                cpu_data = method_data[method_data["device"] == "cpu"]
                cuda_data = method_data[method_data["device"] == "cuda"]

                if not cpu_data.empty and not cuda_data.empty:
                    if cuda_data["mean_time"].values[0] > 0:
                        speedup = (
                            cpu_data["mean_time"].values[0]
                            / cuda_data["mean_time"].values[0]
                        )
                        speedups.append((method, speedup))

            speedups.sort(key=lambda x: x[1], reverse=True)
            for i, (method, speedup) in enumerate(speedups[:5], 1):
                f.write(f"  {i}. {method}: {speedup:.2f}x\n")

        print(f"✅ Результаты сохранены в: {output_dir}")

    def _plot_results(self, df: pd.DataFrame, test_name: str):
        """Визуализация результатов"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(self.base_output_dir, f"{test_name}_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)

        # График 1: Сравнение времени (CPU vs CUDA)
        plt.figure(figsize=(14, 8))

        methods = df["method"].unique()
        x = np.arange(len(methods))
        width = 0.35

        cpu_times = []
        cuda_times = []

        for method in methods:
            method_data = df[df["method"] == method]
            cpu_data = method_data[method_data["device"] == "cpu"]
            cuda_data = method_data[method_data["device"] == "cuda"]

            cpu_times.append(
                cpu_data["mean_time"].values[0] * 1000 if not cpu_data.empty else 0
            )
            cuda_times.append(
                cuda_data["mean_time"].values[0] * 1000 if not cuda_data.empty else 0
            )

        plt.bar(x - width / 2, cpu_times, width, label="CPU", color="#3498db")
        plt.bar(x + width / 2, cuda_times, width, label="CUDA", color="#e74c3c")

        plt.xlabel("Метод сегментации", fontsize=12)
        plt.ylabel("Время выполнения (ms)", fontsize=12)
        plt.title(
            "Сравнение производительности: CPU vs CUDA", fontsize=14, fontweight="bold"
        )
        plt.xticks(x, methods, rotation=45, ha="right", fontsize=9)
        plt.legend(fontsize=11)
        plt.grid(axis="y", alpha=0.3, linestyle="--")
        plt.tight_layout()

        plot_path = os.path.join(output_dir, "cpu_cuda_comparison.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"📊 График сохранён: {plot_path}")

        # График 2: Ускорение (Speedup)
        if torch.cuda.is_available():
            plt.figure(figsize=(12, 6))

            speedups = []
            method_names = []

            for method in methods:
                method_data = df[df["method"] == method]
                cpu_data = method_data[method_data["device"] == "cpu"]
                cuda_data = method_data[method_data["device"] == "cuda"]

                if not cpu_data.empty and not cuda_data.empty:
                    if cuda_data["mean_time"].values[0] > 0:
                        speedup = (
                            cpu_data["mean_time"].values[0]
                            / cuda_data["mean_time"].values[0]
                        )
                        speedups.append(speedup)
                        method_names.append(method)

            if speedups:
                x = np.arange(len(method_names))
                colors = ["#2ecc71" if s > 1 else "#e74c3c" for s in speedups]

                plt.bar(x, speedups, color=colors, edgecolor="black", linewidth=1.2)
                plt.axhline(
                    y=1, color="gray", linestyle="--", linewidth=2, label="CPU = CUDA"
                )

                plt.xlabel("Метод сегментации", fontsize=12)
                plt.ylabel("Ускорение (CPU time / CUDA time)", fontsize=12)
                plt.title(
                    "Ускорение при использовании CUDA", fontsize=14, fontweight="bold"
                )
                plt.xticks(x, method_names, rotation=45, ha="right", fontsize=9)
                plt.legend(fontsize=11)
                plt.grid(axis="y", alpha=0.3, linestyle="--")
                plt.tight_layout()

                speedup_path = os.path.join(output_dir, "speedup_comparison.png")
                plt.savefig(
                    speedup_path, dpi=300, bbox_inches="tight", facecolor="white"
                )
                plt.close()
                print(f"📊 График ускорения сохранён: {speedup_path}")

        # График 3: Scatter plot (CPU vs CUDA время)
        if torch.cuda.is_available():
            plt.figure(figsize=(10, 10))

            cpu_vals = []
            cuda_vals = []
            method_labels = []

            for method in methods:
                method_data = df[df["method"] == method]
                cpu_data = method_data[method_data["device"] == "cpu"]
                cuda_data = method_data[method_data["device"] == "cuda"]

                if not cpu_data.empty and not cuda_data.empty:
                    cpu_vals.append(cpu_data["mean_time"].values[0] * 1000)
                    cuda_vals.append(cuda_data["mean_time"].values[0] * 1000)
                    method_labels.append(method)

            if cpu_vals:
                plt.scatter(
                    cpu_vals, cuda_vals, s=100, alpha=0.7, c="blue", edgecolors="black"
                )

                # Линия равенства
                max_val = max(max(cpu_vals), max(cuda_vals))
                plt.plot(
                    [0, max_val], [0, max_val], "r--", linewidth=2, label="CPU = CUDA"
                )

                # Подписи точек
                for i, method in enumerate(method_labels):
                    plt.annotate(
                        method,
                        (cpu_vals[i], cuda_vals[i]),
                        textcoords="offset points",
                        xytext=(5, 5),
                        fontsize=8,
                        ha="left",
                    )

                plt.xlabel("CPU время (ms)", fontsize=12)
                plt.ylabel("CUDA время (ms)", fontsize=12)
                plt.title(
                    "CPU vs CUDA: Время выполнения", fontsize=14, fontweight="bold"
                )
                plt.legend(fontsize=11)
                plt.grid(True, alpha=0.3, linestyle="--")
                plt.tight_layout()

                scatter_path = os.path.join(output_dir, "cpu_cuda_scatter.png")
                plt.savefig(
                    scatter_path, dpi=300, bbox_inches="tight", facecolor="white"
                )
                plt.close()
                print(f"📊 Scatter plot сохранён: {scatter_path}")
