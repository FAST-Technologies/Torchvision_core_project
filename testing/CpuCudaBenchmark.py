# testing/CpuCudaBenchmark.py

"""Модуль для бенчмаркинга производительности и стабильности методов сегментации на CPU и CUDA.

Предназначен для автоматизированного сравнения времени выполнения, ускорения (speedup)
и стабильности алгоритмов сегментации при запуске на разных устройствах и бэкендах:
- Устройства: CPU, CUDA (GPU)
- Бэкенды: PyTorch, OpenCV, Scikit-learn, ONNX, TensorRT
- Точности: fp32, fp16, bf16

Основные возможности:
- Точный тайминг с `time.perf_counter()` и обязательной `torch.cuda.synchronize()` для GPU
- Warm-up прогоны для стабилизации производительности (избегает искажений из-за ленивой инициализации CUDA)
- Автоматическое переключение устройства сегментера (`.device`, `.model.to()`)
- Устойчивость к ошибкам: падение одного метода не останавливает весь бенчмарк
- Сохранение артефактов: результат сегментации, бинарная маска, overlay-визуализация
- Экспорт результатов: CSV, Excel (с автоформатированием), текстовый отчёт со сводкой
- Генерация визуализаций: сравнительные гистограммы, графики ускорения, scatter-plot зависимости

Примечание:
- Имена методов ожидаются в формате: `{base_method}_{backend}_{precision}`
  (например, `otsu_thresholding_ONNX_fp16`, `canny_edge_CV2_fp32`).
- Для корректного переключения устройства сегментер должен иметь атрибут `.device` типа `torch.device`.
- Бенчмарк не оценивает качество сегментации — только производительность.
  Для валидации качества используйте `BatchClassicTester2` или `SegmentationMetrics`.
"""

# ──────────────────────────────────────────────────────────────────────
# ИМПОРТЫ
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations  # PEP 563: отложенная оценка аннотаций

import os
import sys
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, List
from datetime import datetime
from PIL import Image

import logging

# Настройка логгера
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

project_root: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# ──────────────────────────────────────────────────────────────────────
class CpuCudaBenchmark:
    """Бенчмарк для сравнения производительности и стабильности методов сегментации.

    Использует устройства CPU и CUDA с поддержкой множественных бэкендов (PyTorch/ONNX/TRT)
    и точностей (fp32/fp16/bf16).

    Особенности:
    - Автоматическое переключение устройств для PyTorch-совместимых сегментеров.
    - Warm-up прогоны для стабилизации производительности GPU (избегает искажений из-за
      ленивой инициализации CUDA-контекста и аллокации памяти).
    - Точный тайминг с `time.perf_counter()` и обязательной `torch.cuda.synchronize()`
      для GPU-замеров.
    - Устойчивость к ошибкам: падение одного метода не останавливает весь бенчмарк.
    - Экспорт в CSV, Excel (с автоподбором ширины колонок) и текстовый отчёт.
    - Генерация трёх типов визуализаций: сравнительная гистограмма, график ускорения (speedup)
      и scatter-plot зависимости времени выполнения.

    Workflow:
    1. Инициализация параметров (кол-во прогонов, warm-up, директория вывода).
    2. Для каждого метода: переключение устройства → warm-up → N запусков с замером → статистика.
    3. Агрегация результатов, расчёт ускорения CUDA/CPU.
    4. Сохранение артефактов и визуализация.
    """

    def __init__(
        self,
        base_output_dir: str = "./data/cpu_cuda_benchmark",
        n_runs: int = 5,
        warmup_runs: int = 2,
    ) -> None:
        """Инициализация бенчмарка.

        Args:
            base_output_dir: Базовая директория для сохранения отчётов и графиков.
                Для каждого запуска создаётся подпапка с таймстампом.
            n_runs: Количество основных измерительных прогонов для каждого метода/устройства.
                Результаты усредняются для снижения дисперсии и выбросов.
            warmup_runs: Количество "разогревочных" прогонов перед замерами.
                Критически важны для GPU, чтобы прогреть троттлинг, закэшировать ядра
                и выделить тензорную память до начала измерений.
        """
        self.base_output_dir: str = base_output_dir
        self.n_runs: int = n_runs
        self.warmup_runs: int = warmup_runs

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _parse_method_info(method_name: str) -> Dict[str, str]:
        """Парсит имя метода вида 'otsu_thresholding_ONNX_fp32' в компоненты."""
        parts: List[str] = method_name.split("_")
        precision: str = parts[-1].lower()
        if precision in ["fp32", "fp16", "bf16"] and len(parts) >= 3:
            backend_raw: str = parts[-2]
            base_method: str = "_".join(parts[:-2])
        else:
            # Fallback: Otsu_Thresholding_CV2 -> base=Otsu_Thresholding, backend=CV2, precision=fp32
            precision = "fp32"
            last: str = parts[-1].upper()
            # Убираем суффиксы версий
            if last in ["V1", "V2", "V3"]:
                backend_raw = parts[-2].upper()
                base_method = "_".join(parts[:-2])
            else:
                backend_raw = last
                base_method = "_".join(parts[:-1])
        if "TORCH" in backend_raw or backend_raw in ["PYTORCH", "NEURAL"]:
            backend = "PyTorch"
        elif "CV2" in backend_raw or "OPENCV" in backend_raw:
            backend = "CV2"
        elif "SKLEARN" in backend_raw:
            backend = "Sklearn"
        elif "ONNX" in backend_raw:
            backend = "ONNX"
        elif "TRT" in backend_raw or "TENSORRT" in backend_raw:
            backend = "TRT"
        else:
            backend = "PyTorch"  # Дефолт
        return {"base_method": base_method, "backend": backend, "precision": precision}

    # ──────────────────────────────────────────────────────────────────────
    def benchmark_method(
        self,
        segmenter: Any,
        image: np.ndarray,
        method_name: str,
        device: str = "cpu",
        save_artifacts: bool = True,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Выполняет точный замер времени выполнения одного метода сегментации на указанном устройстве.

        Логика:
        - Временно переключает `.device` сегментера (и `.model`, если присутствует) на целевое.
        - Выполняет `warmup_runs` прогонов без замера.
        - Выполняет `n_runs` прогонов с принудительной синхронизацией CUDA (если `device="cuda"`).
        - Восстанавливает исходное устройство сегментера.
        - Сохраняет результат сегментации, маску и overlay (если save_artifacts=True).
        - При исключениях пропускает прогон, но продолжает цикл.

        Args:
            segmenter: Объект-сегментер, реализующий метод `.segment(image)`.
                Ожидается атрибут `.device` типа `torch.device` для динамического переключения.
            image: Входное изображение в формате `np.ndarray` (любой формы/канальности).
            method_name: Уникальное имя метода для логирования и маркировки результатов.
            device: Целевое устройство выполнения (`"cpu"` или `"cuda"`).
            save_artifacts: Если True, сохранять результат, маску и overlay после сегментации.
            output_dir: Директория для сохранения артефактов. Если None, используется временная.

        Returns:
            Словарь со статистикой производительности:
            - `method`, `device`: Идентификаторы теста.
            - `mean_time`, `std_time`, `min_time`, `max_time`: Статистика времени (в секундах).
            - `total_time`: Общее время замера, включая накладные расходы Python.
            - `n_runs`: Фактическое количество успешных прогонов.
            - `error`: Строка с описанием ошибки, если ни один прогон не выполнился успешно.
        """
        times: List[float] = []
        last_result: Optional[np.ndarray] = None
        last_mask: Optional[np.ndarray] = None

        # Переключение устройства для Torch-сегментеров
        original_device = None
        info: Dict[str, str] = self._parse_method_info(method_name)
        if hasattr(segmenter, "device") and isinstance(segmenter.device, torch.device):
            original_device = segmenter.device
            segmenter.device = torch.device(device)
            if hasattr(segmenter, "model") and segmenter.model is not None:
                try:
                    segmenter.model = segmenter.model.to(device)
                except Exception:
                    pass

        # Warm-up
        actual_warmup: int = max(
            self.warmup_runs,
            5 if getattr(segmenter, "use_compile", False) else self.warmup_runs,
        )
        for _ in range(actual_warmup):
            try:
                segmenter.segment(image)
            except Exception:
                pass

        # Основное тестирование
        (torch.cuda.synchronize() if device == "cuda" and torch.cuda.is_available() else None)
        start_total: float = time.perf_counter()

        for run in range(self.n_runs):
            try:
                if device == "cuda":
                    torch.cuda.synchronize()
                start: float = time.perf_counter()
                if hasattr(segmenter, "segment_with_mask"):
                    result_opt, mask_opt = segmenter.segment_with_mask(image)
                    result = result_opt if result_opt is not None else segmenter.segment(image)
                    mask = mask_opt if mask_opt is not None else result
                else:
                    result = segmenter.segment(image)
                    mask = result
                if device == "cuda":
                    torch.cuda.synchronize()
                end: float = time.perf_counter()
                times.append(end - start)
                if run == 0 and save_artifacts:
                    last_result = result
                    last_mask = mask
            except Exception as e:
                print(f"⚠️ Error in {method_name} ({device}), run {run + 1}: {e}")
                break

        (torch.cuda.synchronize() if device == "cuda" and torch.cuda.is_available() else None)
        end_total: float = time.perf_counter()

        artifact_paths: Dict[str, Optional[str]] = {
            "result": None,
            "mask": None,
            "overlay": None,
        }

        if save_artifacts and last_result is not None and output_dir:
            try:
                os.makedirs(output_dir, exist_ok=True)

                # Конвертация результата в изображение
                result_to_save: np.ndarray = last_result
                if isinstance(result_to_save, torch.Tensor):
                    result_to_save = result_to_save.cpu().numpy()
                if result_to_save.ndim == 2:
                    result_pil = Image.fromarray(result_to_save.astype(np.uint8))
                elif result_to_save.ndim == 3 and result_to_save.shape[0] in [1, 3]:
                    # (C, H, W) -> (H, W, C)
                    result_to_save = np.transpose(result_to_save, (1, 2, 0))
                    result_pil = Image.fromarray(result_to_save.astype(np.uint8))
                else:
                    result_pil = Image.fromarray(result_to_save.astype(np.uint8))

                # Сохранение результата
                result_path: str = os.path.join(output_dir, f"{method_name}_{device}_result.jpg")
                result_pil.save(result_path)
                artifact_paths["result"] = result_path

                # Сохранение маски
                if last_mask is not None:
                    mask_to_save: np.ndarray = last_mask
                    if isinstance(mask_to_save, torch.Tensor):
                        mask_to_save = mask_to_save.cpu().numpy()
                    if mask_to_save.max() <= 1.0:
                        mask_to_save = (mask_to_save * 255).astype(np.uint8)
                    else:
                        mask_to_save = mask_to_save.astype(np.uint8)

                    mask_path: str = os.path.join(output_dir, f"{method_name}_{device}_mask.png")
                    Image.fromarray(mask_to_save).save(mask_path)
                    artifact_paths["mask"] = mask_path

                    # Создание overlay (оригинал + результат)
                    if isinstance(image, np.ndarray):
                        orig_to_save: np.ndarray = image
                        if orig_to_save.ndim == 2:
                            orig_rgb: np.ndarray = np.stack([orig_to_save] * 3, axis=-1)
                        else:
                            orig_rgb = orig_to_save

                        # Приводим результат к 3 каналам если нужно
                        if result_to_save.ndim == 2:
                            result_rgb: np.ndarray = np.stack([result_to_save] * 3, axis=-1)
                        else:
                            result_rgb = result_to_save

                        # Alpha-смешивание: 30% оригинал + 70% результат
                        overlay: np.ndarray = (orig_rgb * 0.3 + result_rgb * 0.7).astype(np.uint8)
                        overlay_path: str = os.path.join(output_dir, f"{method_name}_{device}_overlay.jpg")
                        Image.fromarray(overlay).save(overlay_path)
                        artifact_paths["overlay"] = overlay_path

            except Exception as e:
                print(f"⚠️ Ошибка сохранения артефактов для {method_name}: {e}")

        # Восстановление оригинального устройства
        if original_device is not None:
            segmenter.device = original_device
            if hasattr(segmenter, "model") and segmenter.model is not None:
                try:
                    segmenter.model = segmenter.model.to(original_device)
                except Exception:
                    pass

        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            import gc

            gc.collect()

        if not times:
            return {
                "method": method_name,
                "device": device,
                "error": "Failed/Skipped",
                "mean_time": float("inf"),
                "std_time": 0.0,
                "min_time": float("inf"),
                "max_time": float("inf"),
                "total_time": end_total - start_total,
                "n_runs": 0,
                "artifact_paths": artifact_paths,
                **info,
            }

        return {
            "method": method_name,
            "device": device,
            "error": None,
            "mean_time": np.mean(times),
            "std_time": np.std(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
            "total_time": end_total - start_total,
            "n_runs": len(times),
            "artifact_paths": artifact_paths,
            **info,
        }

    # ──────────────────────────────────────────────────────────────────────
    def benchmark_all_methods(
        self,
        methods_dict: Dict[str, Any],
        image: np.ndarray,
        test_name: str = "multi_backend_comparison",
        save_artifacts: bool = True,
        artifact_base_dir: Optional[str] = None,
    ) -> pd.DataFrame:
        """Запускает полный цикл бенчмарка для набора методов на CPU и (опционально) CUDA.

        Для каждого метода последовательно выполняет замеры на CPU и GPU,
        рассчитывает ускорение (speedup), выводит прогресс в консоль,
        сохраняет результаты и генерирует графики.

        Args:
            methods_dict: Словарь `{имя_метода: экземпляр_сегментера}`.
            image: Тестовое изображение для всех методов.
            test_name: Префикс имени эксперимента для организации выходных файлов.

        Returns:
            `pd.DataFrame` с агрегированными метриками производительности.
        """
        all_results: List = []
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")

        print("\n" + "=" * 80)
        print("🚀 БЕНЧМАРК: MULTI-BACKEND (CPU vs CUDA)")
        print("=" * 80)
        print(f"Количество методов: {len(methods_dict)}")
        print(f"Количество прогонов: {self.n_runs}")
        print(f"Warm-up прогонов: {self.warmup_runs}")
        print(f"Устройства: {', '.join(devices)}")
        print(f"Сохранение артефактов: {'✅ ВКЛ' if save_artifacts else '❌ ВЫКЛ'}")
        print("=" * 80)

        # Сохраняем результаты
        timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir: str = os.path.join(self.base_output_dir, f"{test_name}_{timestamp}")
        if artifact_base_dir is None and save_artifacts:
            artifact_base_dir = os.path.join(output_dir, "artifacts")

        # Создаём директорию для артефактов заранее
        if save_artifacts and artifact_base_dir:
            os.makedirs(artifact_base_dir, exist_ok=True)

        for method_name, segmenter in methods_dict.items():
            print(f"\n🔹 Метод: {method_name}")
            for device in devices:
                if save_artifacts and artifact_base_dir:
                    method_artifact_dir: str = os.path.join(artifact_base_dir, device, method_name)
                    os.makedirs(method_artifact_dir, exist_ok=True)

                if device == "cuda":
                    print(f"   📊 Тестирование на {device.upper()}...")
                    if torch.cuda.is_available():
                        res: Dict[str, Any] = self.benchmark_method(
                            segmenter,
                            image,
                            method_name,
                            device,
                            save_artifacts=save_artifacts,
                            output_dir=method_artifact_dir,
                        )
                    else:
                        res = {"error": "Skipped (No CUDA)"}
                else:
                    print(f"   📊 Тестирование на {device.upper()}...")
                    res = self.benchmark_method(
                        segmenter,
                        image,
                        method_name,
                        device,
                        save_artifacts=save_artifacts,
                        output_dir=method_artifact_dir,
                    )
                all_results.append(res)
                if res["mean_time"] != float("inf"):
                    print(
                        f" {device.upper()}:     {res['mean_time'] * 1000:.2f}ms ± {res['std_time'] * 1000:.2f}ms ({res['n_runs']} runs)"
                    )
                else:
                    print("      ⏭️ Пропущено/Ошибка")

                if res["mean_time"] != float("inf"):
                    time_str: str = f"{res['mean_time'] * 1000:.2f}ms ± {res['std_time'] * 1000:.2f}ms"
                    print(f"   {device}: {time_str} ({res['n_runs']} runs)")

                    # Логируем сохранённые артефакты
                    if save_artifacts and res.get("artifact_paths"):
                        paths = res["artifact_paths"]
                        saved = [k for k, v in paths.items() if v is not None]
                        if saved:
                            print(f"   💾 Сохранено: {', '.join(saved)}")
                else:
                    print("      ⏭️ Пропущено/Ошибка")

            # Ускорение
            cpu_res = next(
                (r for r in all_results if r["method"] == method_name and r["device"] == "cpu"),
                None,
            )
            cuda_res = next(
                (r for r in all_results if r["method"] == method_name and r["device"] == "cuda"),
                None,
            )

            if cpu_res and cuda_res and cpu_res["mean_time"] > 0 and cuda_res["mean_time"] > 0:
                speedup = cpu_res["mean_time"] / cuda_res["mean_time"]
                print(f"      ⚡ Ускорение: {speedup:.2f}x")

        df: pd.DataFrame = pd.DataFrame(all_results)
        df_valid: pd.DataFrame = df[
            (df["error"].isna()) | (df["error"] == "None") | (df["error"] == "Failed/Skipped")
        ].copy()

        self._save_results(df, test_name, output_dir=output_dir)
        self._plot_results(df_valid, test_name, output_dir=output_dir)

        print(f"\n✅ Все результаты сохранены в: {output_dir}")
        if save_artifacts and artifact_base_dir:
            print(f"📁 Артефакты: {artifact_base_dir}")

        return df

    # ──────────────────────────────────────────────────────────────────────
    def _save_results(self, df: pd.DataFrame, test_name: str, output_dir: Optional[str] = None) -> None:
        """Сохраняет результаты бенчмарка в нескольких форматах.

        Создаёт директорию (если не указана), записывает:
        1. `results.csv` — сырые данные для дальнейшего анализа.
        2. `results.xlsx` — таблица с автоподбором ширины колонок через `openpyxl`.
        3. `report.txt` — текстовый отчёт со сводкой, расчётом ускорений и топ-5 лучших методов.

        Args:
            df: Датафрейм с результатами бенчмарка.
            test_name: Имя эксперимента (используется для генерации имени папки).
            output_dir: Путь к директории сохранения. Если `None`, создаётся автоматически
                с таймстампом внутри `self.base_output_dir`.
        """
        if output_dir is None:
            timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(self.base_output_dir, f"benchmark_{test_name}_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)

        # CSV
        csv_path: str = os.path.join(output_dir, "results.csv")
        df.to_csv(csv_path, index=False)

        # Excel с форматированием
        try:
            excel_path: str = os.path.join(output_dir, "results.xlsx")
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
                        except Exception:
                            pass
                    adjusted_width: int = min(max_length + 2, 50)
                    worksheet.column_dimensions[column_letter].width = adjusted_width
        except Exception as e:
            print(f"⚠️ Не удалось сохранить Excel: {e}")

        # Текстовый отчёт
        report_path: str = os.path.join(output_dir, "report.txt")
        is_multi_backend: bool = "backend" in df.columns and "base_method" in df.columns
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 90 + "\n")
            f.write("📊 ОТЧЁТ: СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ (CPU vs CUDA | Multi-Backend)\n")
            f.write("=" * 90 + "\n")
            f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Количество методов: {df['method'].nunique()}\n")
            f.write(f"Прогонов: {self.n_runs} | Warm-up: {self.warmup_runs}\n")
            f.write("=" * 90 + "\n\n")

            # ──────────────────────────────────────────────────────────────
            # БЛОК 1: Multi-Backend анализ (если есть колонки)
            # ──────────────────────────────────────────────────────────────
            if is_multi_backend:
                # Находим референс: PyTorch / fp32 / CPU
                ref_mask: pd.Series = (
                    (df["backend"] == "PyTorch")
                    & (df["precision"] == "fp32")
                    & (df["device"] == "cpu")
                    & ((df["error"].isna()) | (df["error"] == "None"))
                    & (df["mean_time"] > 0)
                )

                ref_time: Optional[float] = None
                if ref_mask.any():
                    # Берём среднее время всех PyTorch fp32 CPU методов как стабильный референс
                    ref_time = df.loc[ref_mask, "mean_time"].mean()
                    print(f"✅ Референс найден (PyTorch/fp32/CPU): {ref_time * 1000:.2f}ms")
                    f.write(f"Current ref mask: {ref_mask}")
                    f.write(f"Current ref time: {ref_time}")
                else:
                    # 🆘 Fallback: ищем любой валидный CPU метод
                    cpu_valid: pd.DataFrame = df[
                        (df["device"] == "cpu")
                        & (df["mean_time"] > 0)
                        & ((df["error"].isna()) | (df["error"] == "None"))
                    ]
                    if not cpu_valid.empty:
                        # Медиана устойчива к сверхбыстрым CV2 методам (0.3ms) и сверхмедленным Sklearn (40ms)
                        ref_time = cpu_valid["mean_time"].median()
                        ref_method = cpu_valid.loc[cpu_valid["mean_time"].idxmin(), "method"]
                        f.write(
                            f"\n⚠️ PyTorch/fp32/CPU не найден. Fallback (медиана CPU): {ref_time * 1000:.2f}ms (от {ref_method})\n"
                        )

                f.write("🔹 MULTI-BACKEND ANALYSIS:\n" + "-" * 90 + "\n")
                valid_base_methods: List[Any] = [m for m in df["base_method"].unique() if pd.notna(m)]
                for base_method in sorted(valid_base_methods):
                    f.write(f"🔸 БАЗОВЫЙ МЕТОД: {base_method}\n")
                    f.write(
                        f"{'Backend':<12} | {'Precision':<9} | {'Device':<6} | {'Время (мс)':<20} | {'Speedup (vs Ref)':<15}\n"
                        + "-" * 90
                        + "\n"
                    )

                    subset = df[df["base_method"] == base_method].copy()

                    cpu_data = subset[subset["device"] == "cpu"]
                    if not cpu_data.empty:
                        f.write(
                            f"  CPU:  {cpu_data['mean_time'].values[0] * 1000:.2f}ms ± {cpu_data['std_time'].values[0] * 1000:.2f}ms\n"
                        )

                    cuda_data = subset[subset["device"] == "cuda"]
                    if not cuda_data.empty:
                        f.write(
                            f"  CUDA: {cuda_data['mean_time'].values[0] * 1000:.2f}ms ± {cuda_data['std_time'].values[0] * 1000:.2f}ms\n"
                        )

                        if not cpu_data.empty and cuda_data["mean_time"].values[0] > 0:
                            speedup: float = cpu_data["mean_time"].values[0] / cuda_data["mean_time"].values[0]
                            f.write(f"  ⚡ Ускорение: {speedup:.2f}x\n")
                    for _, row in subset.iterrows():
                        is_success = pd.isna(row["error"])
                        if is_success and pd.notna(row["mean_time"]) and row["mean_time"] > 0:
                            time_ms: float = row["mean_time"] * 1000
                            std_ms: float = row["std_time"] * 1000
                            time_str: str = f"{time_ms:.2f} ± {std_ms:.2f}"

                            speedup_str: str = "N/A"
                            if ref_time and ref_time > 0:
                                spd: float = ref_time / row["mean_time"]
                                speedup_str = f"{spd:.2f}x {'⚡' if spd > 1.0 else '🐢'}"
                        else:
                            time_str = "N/A"
                            speedup_str = "N/A"

                        f.write(
                            f"{row['backend']:<12} | {row['precision']:<9} | {row['device']:<6} | {time_str:<20} | {speedup_str:<15}\n"
                        )
                    f.write("\n")

                # Топ ускорений относительно референса
                if ref_time:
                    f.write("=" * 90 + "\n🏆 ТОП-5 УСКОРЕНИЙ (относительно PyTorch/fp32/CPU):\n" + "=" * 90 + "\n")
                    valid: pd.DataFrame = df[
                        (df["mean_time"] != float("inf")) & (df["mean_time"] > 0) & (df["error"].isna())
                    ]
                    speedups = []
                    for _, r in valid.iterrows():
                        if not (r["backend"] == "PyTorch" and r["precision"] == "fp32" and r["device"] == "cpu"):
                            speedups.append(
                                (
                                    f"{r['base_method']} / {r['backend']} / {r['precision']} / {r['device']}",
                                    ref_time / r["mean_time"],
                                )
                            )
                    for i, (name, spd) in enumerate(sorted(speedups, key=lambda x: x[1], reverse=True)[:20], 1):
                        f.write(f"  {i}. {name}: {spd:.2f}x\n")
                    f.write("\n")

            # ──────────────────────────────────────────────────────────────
            # БЛОК 2: Legacy CPU vs CUDA summary (обратная совместимость)
            # ──────────────────────────────────────────────────────────────
            if "device" in df.columns:
                speedups_legacy: List = []
                f.write("🔹 LEGACY CPU vs CUDA SUMMARY:\n" + "-" * 90 + "\n")
                for method in sorted(df["method"].unique()):
                    m_data = df[df["method"] == method]
                    cpu = m_data[m_data["device"] == "cpu"]
                    cuda = m_data[m_data["device"] == "cuda"]

                    if cpu.empty or cuda.empty:
                        continue

                    cpu_t: float = cpu["mean_time"].values[0] * 1000
                    cuda_t: float = (
                        cuda["mean_time"].values[0] * 1000 if cuda["mean_time"].values[0] > 0 else float("inf")
                    )
                    spd = cpu_t / cuda_t if cuda_t > 0 else float("inf")
                    speedups_legacy.append((method, spd))

                    f.write(f"{method:<40s}: CPU={cpu_t:7.2f}ms | CUDA={cuda_t:7.2f}ms | ⚡ Speedup={spd:.2f}x\n")

            speedups_legacy.sort(key=lambda x: x[0], reverse=True)
            for i, (method, speedup) in enumerate(speedups_legacy[:30], 1):
                f.write(f"  {i}. {method}: {speedup:.2f}x\n")

        print(f"✅ Результаты сохранены в: {output_dir}/report.txt")

    # ──────────────────────────────────────────────────────────────────────
    def _plot_results(self, df: pd.DataFrame, test_name: str, output_dir: Optional[str] = None) -> None:
        """Генерирует и сохраняет визуализации производительности.

        Создаёт три графика:
        1. `cpu_cuda_comparison.png` — группированная гистограмма среднего времени (ms).
        2. `speedup_comparison.png` — бар-чарт ускорения (CPU_time / CUDA_time).
           Зелёные столбцы: ускорение > 1x, красные: замедление < 1x.
        3. `cpu_cuda_scatter.png` — scatter-plot зависимости CUDA времени от CPU времени.
           Красная пунктирная линия обозначает линию равенства производительности.

        Args:
            df: Датафрейм с результатами бенчмарка.
            test_name: Имя эксперимента.
            output_dir: Директория для сохранения графиков. Генерируется автоматически, если `None`.
        """
        if df.empty:
            return
        if output_dir is None:
            timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(self.base_output_dir, f"{test_name}_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)

        def _parse_speedup(val: str) -> float:
            """Извлекает числовое значение из строки speedup (например, '1.5×' → 1.5)."""
            if isinstance(val, (int, float)):
                return float(val)
            try:
                # Удаляем символы "×", "x", пробелы и приводим к float
                return float(str(val).replace("×", "").replace("x", "").strip())
            except (ValueError, AttributeError):
                return 1.0  # Default при ошибке парсинга

        # ──────────────────────────────────────────────────────────────
        # ГРАФИК 1: Legacy CPU vs CUDA Bar
        # ──────────────────────────────────────────────────────────────
        plt.figure(figsize=(14, 8))

        methods = df["method"].unique()
        x = np.arange(len(methods))
        width: float = 0.35

        cpu_times: List[float] = []
        cuda_times: List[float] = []

        for method in methods:
            method_data = df[df["method"] == method]
            cpu_data = method_data[method_data["device"] == "cpu"]
            cuda_data = method_data[method_data["device"] == "cuda"]

            cpu_times.append(cpu_data["mean_time"].values[0] * 1000 if not cpu_data.empty else 0)
            cuda_times.append(cuda_data["mean_time"].values[0] * 1000 if not cuda_data.empty else 0)

        plt.bar(x - width / 2, cpu_times, width, label="CPU", color="#3498db")
        plt.bar(x + width / 2, cuda_times, width, label="CUDA", color="#e74c3c")

        plt.xlabel("Метод сегментации", fontsize=12)
        plt.ylabel("Время выполнения (ms)", fontsize=12)
        plt.title("Сравнение производительности: CPU vs CUDA", fontsize=14, fontweight="bold")
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

            speedups: List[str] = []
            method_names: List[str] = []

            for method in methods:
                method_data = df[df["method"] == method]
                cpu_data = method_data[method_data["device"] == "cpu"]
                cuda_data = method_data[method_data["device"] == "cuda"]

                if not cpu_data.empty and not cuda_data.empty:
                    if cuda_data["mean_time"].values[0] > 0:
                        speedup = cpu_data["mean_time"].values[0] / cuda_data["mean_time"].values[0]
                        speedups.append(speedup)
                        method_names.append(method)

            if speedups:
                x = np.arange(len(method_names))

                # Используем функцию для безопасного сравнения
                colors: List[str] = ["#2ecc71" if _parse_speedup(s) > 1.0 else "#e74c3c" for s in speedups]

                plt.bar(x, speedups, color=colors, edgecolor="black", linewidth=1.2)
                plt.axhline(y=1, color="gray", linestyle="--", linewidth=2, label="CPU = CUDA")

                plt.xlabel("Метод сегментации", fontsize=12)
                plt.ylabel("Ускорение (CPU time / CUDA time)", fontsize=12)
                plt.title("Ускорение при использовании CUDA", fontsize=14, fontweight="bold")
                plt.xticks(x, method_names, rotation=45, ha="right", fontsize=9)
                plt.legend(fontsize=11)
                plt.grid(axis="y", alpha=0.3, linestyle="--")
                plt.tight_layout()

                speedup_path: str = os.path.join(output_dir, "speedup_comparison.png")
                plt.savefig(speedup_path, dpi=300, bbox_inches="tight", facecolor="white")
                plt.close()
                print(f"📊 График ускорения сохранён: {speedup_path}")

        # График 3: Scatter plot (CPU vs CUDA время)
        if torch.cuda.is_available():
            plt.figure(figsize=(10, 10))

            cpu_vals: List = []
            cuda_vals: List = []
            method_labels: List[str] = []

            for method in methods:
                method_data = df[df["method"] == method]
                cpu_data = method_data[method_data["device"] == "cpu"]
                cuda_data = method_data[method_data["device"] == "cuda"]

                if not cpu_data.empty and not cuda_data.empty:
                    cpu_vals.append(cpu_data["mean_time"].values[0] * 1000)
                    cuda_vals.append(cuda_data["mean_time"].values[0] * 1000)
                    method_labels.append(method)

            if cpu_vals:
                plt.scatter(cpu_vals, cuda_vals, s=100, alpha=0.7, c="blue", edgecolors="black")

                # Линия равенства
                max_val = max(max(cpu_vals), max(cuda_vals))
                plt.plot([0, max_val], [0, max_val], "r--", linewidth=2, label="CPU = CUDA")

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
                plt.title("CPU vs CUDA: Время выполнения", fontsize=14, fontweight="bold")
                plt.legend(fontsize=11)
                plt.grid(True, alpha=0.3, linestyle="--")
                plt.tight_layout()

                scatter_path: str = os.path.join(output_dir, "cpu_cuda_scatter.png")
                plt.savefig(scatter_path, dpi=300, bbox_inches="tight", facecolor="white")
                plt.close()
                print(f"📊 Scatter plot сохранён: {scatter_path}")

        # ──────────────────────────────────────────────────────────────
        # ГРАФИК 4: Multi-Backend CUDA Comparison (если есть backend)
        # ─────────────────────────────────────────────────────────────
        if "backend" in df.columns and "base_method" in df.columns:
            cuda_df: pd.DataFrame = df[df["device"] == "cuda"].copy()
            if not cuda_df.empty:
                plt.figure(figsize=(16, 8))
                valid_base = [m for m in cuda_df["base_method"].dropna().unique() if pd.notna(m)]
                base_methods = sorted(valid_base)
                x = np.arange(len(base_methods))
                width = 0.2
                valid_backends: List[str] = [b for b in cuda_df["backend"].dropna().unique() if pd.notna(b)]
                backends = sorted(valid_backends)
                offsets = {b: i for i, b in enumerate(backends)}

                for backend in backends:
                    b_df = cuda_df[cuda_df["backend"] == backend]
                    means = [
                        (
                            b_df[b_df["base_method"] == m]["mean_time"].mean() * 1000
                            if m in b_df["base_method"].values
                            else 0
                        )
                        for m in base_methods
                    ]
                    plt.bar(
                        x + offsets[backend] * width,
                        means,
                        width,
                        label=backend,
                        alpha=0.85,
                    )

                plt.xlabel("Базовый метод", fontsize=12)
                plt.ylabel("Время выполнения (ms) ↓", fontsize=12)
                plt.title(
                    "Сравнение времени выполнения (CUDA, разные бэкенды)",
                    fontsize=14,
                    fontweight="bold",
                )
                plt.xticks(
                    x + width * (len(backends) - 1) / 2,
                    base_methods,
                    rotation=45,
                    ha="right",
                    fontsize=9,
                )
                plt.legend()
                plt.grid(axis="y", alpha=0.3, linestyle="--")
                plt.tight_layout()
                plt.savefig(
                    os.path.join(output_dir, "comparison_bar.png"),
                    dpi=200,
                    bbox_inches="tight",
                )
                plt.close()

        # ──────────────────────────────────────────────────────────────
        # ГРАФИК 5: Speedup относительно референса (если есть backend)
        # ──────────────────────────────────────────────────────────────
        if "backend" in df.columns:
            ref_mask: pd.Series = (
                (df["backend"] == "PyTorch")
                & (df["precision"] == "fp32")
                & (df["device"] == "cpu")
                & (df["error"].isna())
            )
            ref_row: pd.DataFrame = df[ref_mask]
            if not ref_row.empty:
                ref_time: float = ref_row.iloc[0]["mean_time"]
                df_plot: pd.DataFrame = df.copy()
                df_plot["speedup"] = np.where(df_plot["mean_time"] > 0, ref_time / df_plot["mean_time"], np.nan)

                plt.figure(figsize=(14, 7))
                valid: pd.DataFrame = df_plot[(df_plot["speedup"].notna()) & (df_plot["device"] == "cuda")].copy()
                valid = valid.sort_values("speedup", ascending=True)
                if not valid.empty:
                    colors = ["#2ecc71" if s > 1 else "#e74c3c" for s in valid["speedup"]]
                    plt.barh(
                        valid["method"],
                        valid["speedup"],
                        color=colors,
                        edgecolor="black",
                        linewidth=0.8,
                    )
                    plt.axvline(1, color="gray", linestyle="--", linewidth=2)
                    plt.xlabel("Speedup (CPU_ref / CUDA_time)", fontsize=12)
                    plt.title(
                        "Ускорение относительно PyTorch/fp32/CPU",
                        fontsize=14,
                        fontweight="bold",
                    )
                    plt.grid(axis="x", alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(
                        os.path.join(output_dir, "speedup_chart.png"),
                        dpi=200,
                        bbox_inches="tight",
                    )
                    plt.close()

        print(f"📊 Графики сохранены в: {output_dir}")
