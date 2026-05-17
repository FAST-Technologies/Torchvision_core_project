# utils/backend_exporter_new.py
"""Расширенный экспорт нейросетевых моделей в ONNX и TensorRT.

Проблема: torch_tensorrt.compile через JIT/Dynamo падает для SMP-моделей
(PSPNet, U-Net, FPN, SegNet) из-за:
  - aten::size() + dynamic reshape в SMP-декодерах → SIZE_MAX overflow в TRT
  - tuple(int,int,int,int) в padding → Expected ivalue->isInt()
  - Сложные skip-connections с conditional interpolation

Решение: три стратегии в порядке приоритета:
  1. ONNX → TRT через tensorrt Python API (trtexec-эквивалент)
  2. ONNX → TRT через polygraphy/onnx-tensorrt
  3. ONNX Runtime с CUDAExecutionProvider как high-performance fallback

Стратегия 1 обходит все проблемы с torch_tensorrt, т.к. работает с
уже сериализованным ONNX графом, где все dynamic shapes уже разрешены.
"""

from __future__ import annotations

import os
import gc
import time
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Literal

import torch
import torch.nn as nn
import numpy as np

# ──────────────────────────────────────────────────────────────────────
# TYPE ALIASES
# ──────────────────────────────────────────────────────────────────────
ShapeType = Tuple[int, ...]
"""Тип для формы тензора, например (1, 3, 512, 512)."""

OnnxProvider = Union[str, Tuple[str, Dict[str, Any]]]
"""Тип провайдера ONNX Runtime: либо строка, либо (имя, опции)."""

logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler: logging.StreamHandler = logging.StreamHandler()
    formatter: logging.Formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def _check_import(module_name: str) -> bool:
    """Проверка доступности модуля."""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


def _diagnose_onnx_cuda() -> str:
    """Диагностика доступности CUDA в ONNX Runtime."""
    try:
        import onnxruntime as ort

        providers: List[str] = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            return f"✅ CUDA EP available (providers: {providers})"
        else:
            return (
                f"⚠️ CUDA EP missing. Available: {providers}. "
                f"Check: pip install onnxruntime-gpu, CUDA/cuDNN compatible"
            )
    except ImportError:
        return "❌ onnxruntime not installed"


def _should_rebuild_trt(onnx_path: Path, trt_path: Path) -> bool:
    """Проверка, нужно ли пересобирать TRT engine."""
    if not trt_path.exists():
        return True
    return onnx_path.stat().st_mtime > trt_path.stat().st_mtime


def _get_trt_ir_mode() -> Literal["dynamo", "ts"]:
    """Определяет режим IR для torch_tensorrt по версии."""
    try:
        import torch_tensorrt

        version: Tuple[int, ...] = tuple(map(int, torch_tensorrt.__version__.split(".")[:2]))
        logging.info(f"Current torch_tensorrt version: {version}")
        # return "dynamo" if version >= (2, 0) else "ts"
        return "ts"
    except ImportError:
        return "ts"


# Глобальный чеклист зависимостей
CHECKLIST: Dict[str, bool] = {
    "cuda_available": torch.cuda.is_available(),
    "tensorrt_api": _check_import("tensorrt"),
    "onnxruntime_gpu": _check_import("onnxruntime")
    and "CUDAExecutionProvider" in __import__("onnxruntime").get_available_providers(),
    "torch_tensorrt": _check_import("torch_tensorrt"),
    "onnx_simplifier": _check_import("onnxsim"),
}


# ─────────────────────────────────────────────────────────────────────
# Стратегия 1: ONNX → TensorRT через Python API (tensorrt напрямую)
# ─────────────────────────────────────────────────────────────────────
def export_onnx_to_trt_via_api(
    onnx_path: Union[str, Path],
    trt_path: Union[str, Path],
    precision: Literal["fp32", "fp16", "bf16"] = "fp32",
    input_shape: Tuple[int, int, int, int] = (1, 3, 512, 512),
    workspace_gb: float = 4.0,
    verbose: bool = False,
) -> bool:
    """Конвертирует ONNX → TensorRT engine через tensorrt Python API.

    Это эквивалент команды:
        trtexec --onnx=model.onnx --saveEngine=model.trt --fp16

    Преимущество перед torch_tensorrt: работает с любым валидным ONNX,
    не зависит от torch JIT/dynamo ограничений.

    Args:
        onnx_path: Путь к .onnx файлу.
        trt_path: Путь для сохранения .trt engine.
        precision: "fp32", "bf16" или "fp16".
        input_shape: (B, C, H, W) — форма входа.
        workspace_gb: Размер workspace TRT в ГБ.
        verbose: Подробное логирование TRT builder.

    Returns:
        bool: True при успехе.
    """
    try:
        import tensorrt as trt
    except ImportError:
        logger.warning("tensorrt Python API не установлен. Попробуйте: pip install tensorrt")
        return False

    onnx_p: Path = Path(onnx_path)
    trt_p: Path = Path(trt_path)
    if not _should_rebuild_trt(onnx_p, trt_p):
        logger.info(f"⏭ TRT engine актуален, пропускаем сборку: {trt_p}")
        return True

    TRT_LOGGER: trt.Logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)

    try:
        builder: trt.Builder = trt.Builder(TRT_LOGGER)
        network_flags: int = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network: trt.INetworkDefinition = builder.create_network(network_flags)
        parser: trt.OnnxParser = trt.OnnxParser(network, TRT_LOGGER)

        # Парсим ONNX
        with open(onnx_path, "rb") as f:
            onnx_bytes: bytes = f.read()

        if not parser.parse(onnx_bytes):
            errors: List[str] = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            logger.error(f"ONNX parse errors: {errors}")
            return False

        logger.info(
            f"ONNX parsed: {network.num_layers} layers, "
            f"input={network.get_input(0).shape if network.num_inputs > 0 else 'N/A'}"
        )

        # Конфигурация builder
        config: trt.IBuilderConfig = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * 1024**3))

        if precision == "fp16":
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                logger.info("FP16 enabled")
            else:
                logger.warning("GPU не поддерживает fast FP16, используем FP32")

        if precision == "bf16":
            # Проверка совместимости: атрибут может отсутствовать в старых версиях TRT
            has_bf16 = hasattr(builder, "platform_has_fast_bf16") and builder.platform_has_fast_bf16
            if has_bf16:
                config.set_flag(trt.BuilderFlag.BF16)
                logger.info("BF16 enabled")
            else:
                logger.warning("⚠️ BF16 не поддерживается данной версией TensorRT, пробуем FP16 как fallback")
                if builder.platform_has_fast_fp16:
                    config.set_flag(trt.BuilderFlag.FP16)
                    logger.info("FP16 enabled as fallback")

        # Optimization profile для динамических размеров
        profile: trt.IOptimizationProfile = builder.create_optimization_profile()
        input_tensor: trt.ITensor = network.get_input(0)
        input_name: str = input_tensor.name
        onnx_input_shape: Tuple[int, ...] = tuple(input_tensor.shape)
        is_static_input: bool = all(d > 0 for d in onnx_input_shape)

        b, c, h, w = input_shape

        if is_static_input:
            # 🔥 Для статического входа: min=opt=max=onnx_shape
            profile.set_shape(
                input_name,
                min=onnx_input_shape,
                opt=onnx_input_shape,
                max=onnx_input_shape,
            )
            logger.info(f"✅ Static input: profile locked to {onnx_input_shape}")
        else:
            # Для динамического: используем заданные границы
            h_min = max(32, h // 2)
            w_min = max(32, w // 2)
            h_max = min(h * 2, 4096)
            w_max = min(w * 2, 4096)

            profile.set_shape(
                input_name,
                min=(b, c, h_min, w_min),
                opt=(b, c, h, w),
                max=(b, c, h_max, w_max),
            )
            logger.info(f"✅ Dynamic input: profile [{h_min}x{w_min} → {h}x{w} → {h_max}x{w_max}]")
        config.add_optimization_profile(profile)

        # Сборка engine
        logger.info(f"Building TRT engine (precision={precision}, workspace={workspace_gb}GB)...")
        t0: float = time.time()

        serialized_engine: Optional[bytes] = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            logger.error("TRT build_serialized_network вернул None")
            return False

        build_time: float = time.time() - t0
        logger.info(f"Engine built in {build_time:.1f}s")

        # Сохранение
        trt_p.parent.mkdir(parents=True, exist_ok=True)
        with open(trt_p, "wb") as f:
            f.write(serialized_engine)

        engine_size_mb: float = trt_p.stat().st_size / 1e6
        logger.info(f"✅ TRT via tensorrt API engine saved: {trt_p} ({engine_size_mb:.1f} MB)")
        return True

    except Exception as e:
        logger.error(f"❌ ONNX→TRT via API failed: {e}")
        if trt_p.exists():
            trt_p.unlink()
        return False


# ─────────────────────────────────────────────────────────────────────
# Стратегия 2: ONNX → TRT через trtexec subprocess
# ─────────────────────────────────────────────────────────────────────
def export_onnx_to_trt_via_trtexec(
    onnx_path: Union[str, Path],
    trt_path: Union[str, Path],
    precision: Literal["fp32", "fp16", "bf16"] = "fp32",
    input_shape: Tuple[int, int, int, int] = (1, 3, 512, 512),
    workspace_mb: int = 4096,
    trtexec_path: str = "trtexec",
) -> bool:
    """Конвертирует ONNX → TRT через вызов trtexec в subprocess.

    Используется как fallback если tensorrt Python API недоступен.
    trtexec обычно доступен при установке TensorRT через deb/rpm пакеты.

    Args:
        onnx_path: Путь к .onnx.
        trt_path: Путь для .trt.
        precision: "fp32", "bf16" или "fp16".
        input_shape: (B, C, H, W).
        workspace_mb: Workspace в МБ.
        trtexec_path: Путь к trtexec (или просто "trtexec" если в PATH).

    Returns:
        bool: True при успехе.
    """
    import shutil

    onnx_p: Path = Path(onnx_path)
    trt_p: Path = Path(trt_path)

    if not _should_rebuild_trt(onnx_p, trt_p):
        logger.info(f"⏭ TRT engine актуален, пропускаем сборку: {trt_p}")
        return True

    if not shutil.which(trtexec_path):
        logger.warning(f"trtexec не найден в PATH: {trtexec_path}")
        return False

    b, c, h, w = input_shape
    shape_str: str = f"{b}x{c}x{h}x{w}"
    min_shape: str = f"{b}x{c}x{h // 2}x{w // 2}"
    max_shape: str = f"{b}x{c}x{h * 2}x{w * 2}"

    cmd: List[str] = [
        trtexec_path,
        f"--onnx={onnx_path}",
        f"--saveEngine={trt_path}",
        f"--workspace={workspace_mb}",
        f"--minShapes=input:{min_shape}",
        f"--optShapes=input:{shape_str}",
        f"--maxShapes=input:{max_shape}",
    ]

    if precision == "fp16":
        cmd.append("--fp16")

    if precision == "bf16":
        cmd.append("--bf16")  # вместо --fp16

    logger.info(f"Running: {' '.join(cmd)}")

    try:
        result: subprocess.CompletedProcess[str] = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 минут
        )
        if result.returncode == 0:
            logger.info(f"✅ trtexec succeeded: {trt_path}")
            return True
        else:
            logger.error(f"❌ trtexec failed:\n{result.stderr[-2000:]}")
            return False
    except subprocess.TimeoutExpired:
        logger.error("trtexec timeout (>300s)")
        return False
    except Exception as e:
        logger.error(f"trtexec subprocess error: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────
# Стратегия 2.5: ONNX → TRT через onnx-tensorrt parser
# ─────────────────────────────────────────────────────────────────────
def export_onnx_to_trt_via_onnx_tensorrt(
    onnx_path: Union[str, Path],
    trt_path: Union[str, Path],
    precision: Literal["fp32", "fp16", "bf16"] = "fp32",
    input_shape: Tuple[int, int, int, int] = (1, 3, 512, 512),
    workspace_mb: int = 4096,
    verbose: bool = False,
) -> bool:
    """Конвертирует ONNX → TensorRT через onnx-tensorrt parser.

    Использует официальный парсер NVIDIA для конвертации ONNX графа
    в TensorRT engine. Может работать лучше torch_tensorrt для некоторых
    моделей с нестандартными операторами.

    Требует установленного пакета: pip install onnx-tensorrt

    Args:
        onnx_path: Путь к .onnx файлу.
        trt_path: Путь для сохранения .trt engine.
        precision: "fp32", "bf16" или "fp16".
        input_shape: (B, C, H, W) — форма входа.
        workspace_mb: Размер workspace в МБ.
        verbose: Подробное логирование.

    Returns:
        bool: True при успехе.
    """
    try:
        import tensorrt as trt
        import onnx_tensorrt.backend as backend  # noqa: F401
    except ImportError:
        logger.warning(
            "onnx-tensorrt не установлен. Попробуйте: " "pip install onnx-tensorrt (требует tensorrt и onnx)"
        )
        return False

    onnx_p: Path = Path(onnx_path)
    trt_p: Path = Path(trt_path)

    if not _should_rebuild_trt(onnx_p, trt_p):
        logger.info(f"⏭ TRT engine актуален, пропускаем сборку: {trt_p}")
        return True

    TRT_LOGGER: trt.Logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)

    try:
        # Создаём builder и network
        builder: trt.Builder = trt.Builder(TRT_LOGGER)
        network_flags: int = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network: trt.INetworkDefinition = builder.create_network(network_flags)

        # Парсим ONNX через onnx-tensorrt
        parser: trt.OnnxParser = trt.OnnxParser(network, TRT_LOGGER)
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                errors: List[str] = [str(parser.get_error(i)) for i in range(parser.num_errors)]
                logger.error(f"ONNX parse errors: {errors}")
                return False

        # Конфигурация builder
        config: trt.IBuilderConfig = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_mb * 1024**2))

        if precision == "fp16" and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("FP16 enabled")

        # Optimization profile для динамических размеров
        profile: trt.IOptimizationProfile = builder.create_optimization_profile()
        input_tensor: trt.ITensor = network.get_input(0)
        input_name: str = input_tensor.name
        b, c, h, w = input_shape

        h_min = max(32, h // 2)
        w_min = max(32, w // 2)
        h_max = min(h * 2, 4096)
        w_max = min(w * 2, 4096)

        profile.set_shape(
            input_name,
            min=(b, c, h_min, w_min),
            opt=(b, c, h, w),
            max=(b, c, h_max, w_max),
        )
        config.add_optimization_profile(profile)

        # Сборка engine
        logger.info(f"Building TRT engine via onnx-tensorrt (precision={precision})...")
        t0: float = time.time()

        serialized_engine: Optional[bytes] = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            logger.error("TRT build_serialized_network вернул None")
            return False

        build_time: float = time.time() - t0
        logger.info(f"Engine built in {build_time:.1f}s")

        # Сохранение
        trt_p.parent.mkdir(parents=True, exist_ok=True)
        with open(trt_p, "wb") as f:
            f.write(serialized_engine)

        engine_size_mb: float = trt_p.stat().st_size / 1e6
        logger.info(f"✅ TRT via onnx-tensorrt saved: {trt_p} ({engine_size_mb:.1f} MB)")
        return True

    except Exception as e:
        logger.error(f"❌ ONNX→TRT via onnx-tensorrt failed: {e}")
        if trt_p.exists():
            trt_p.unlink()
        return False


# ─────────────────────────────────────────────────────────────────────
# Стратегия 3: ONNX Runtime с CUDAExecutionProvider
# ─────────────────────────────────────────────────────────────────────
class OnnxTrtFallbackSegmenter:
    """Обёртка для инференса через ONNX Runtime с CUDAExecutionProvider.

    Производительность: ~2-5x медленнее нативного TRT, но работает
    для всех архитектур (SMP, PSPNet, SegNet и т.д.).

    Используется когда TRT engine не удалось собрать.
    Совместим с интерфейсом nn.Module: callable, поддерживает .eval().
    """

    def __init__(
        self,
        onnx_path: Union[str, Path],
        input_shape: Tuple[int, int, int, int] = (1, 3, 512, 512),
        device: str = "cuda",
    ) -> None:
        """Инициализация модуля OnnxTrtFallbackSegmenter."""
        try:
            import onnxruntime as ort

            if "CUDAExecutionProvider" not in ort.get_available_providers():
                logger.warning("⚠️ onnxruntime-gpu установлен, но CUDA не доступна. Проверьте CUDA/cuDNN версии.")
                logger.warning(f"⚠️ {_diagnose_onnx_cuda()}")
        except ImportError:
            raise ImportError("pip install onnxruntime-gpu")

        self.input_shape: ShapeType = input_shape

        providers: List[OnnxProvider] = (
            [
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": 0,
                        "arena_extend_strategy": "kNextPowerOfTwo",
                        "gpu_mem_limit": 4 * 1024**3,
                        "cudnn_conv_algo_search": "EXHAUSTIVE",
                        "do_copy_in_default_stream": True,
                    },
                ),
                "CPUExecutionProvider",
            ]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )

        sess_options: ort.SessionOptions = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.enable_mem_pattern = True

        self.session: ort.InferenceSession = ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_options,
            providers=providers,
        )
        self.input_name: str = self.session.get_inputs()[0].name
        self.output_name: str = self.session.get_outputs()[0].name

        actual_providers: List[str] = self.session.get_providers()
        logger.info(f"ONNX Runtime providers: {actual_providers}")
        if "CUDAExecutionProvider" in actual_providers:
            logger.info("✅ CUDA acceleration active")
            cuda_opts = self.session.get_provider_options().get("CUDAExecutionProvider", {})
            logger.debug(f"CUDA options: {cuda_opts}")
        else:
            logger.warning(
                "⚠️ CUDA provider не активен, используется CPU. "
                "Проверьте: 1) onnxruntime-gpu установлен, 2) CUDA доступна"
            )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Инференс: torch.Tensor → torch.Tensor."""
        x_np: np.ndarray = x.cpu().float().numpy()
        outputs: List[np.ndarray] = self.session.run([self.output_name], {self.input_name: x_np})
        result: torch.Tensor = torch.from_numpy(outputs[0]).float()
        if x.is_cuda:
            result = result.cuda()
        return result

    def eval(self) -> "OnnxTrtFallbackSegmenter":
        """Возвращает созданный OnnxTrtFallbackSegmenter."""
        return self  # совместимость с nn.Module интерфейсом


# ─────────────────────────────────────────────────────────────────────
# Основная функция: экспорт нейронной модели в ONNX + TRT
# ─────────────────────────────────────────────────────────────────────
def export_neural_model(
    model: nn.Module,
    model_key: str,
    output_dir: Union[str, Path],
    input_shape: Tuple[int, int, int, int] = (1, 3, 512, 512),
    opset_version: int = 17,
    trt_precision: Literal["fp32", "bf16", "fp16"] = "fp32",
    device: str = "cuda",
    dynamic_axes: bool = False,
) -> Dict[str, Optional[Path]]:
    """Полный pipeline экспорта нейронной модели: PyTorch → ONNX → TRT.

    TRT экспорт пробует ЧЕТЫРЕ стратегии в порядке приоритета:
      1. tensorrt Python API (самый надёжный, работает с любым ONNX)
      2. trtexec subprocess (если Python API недоступен)
      2.5. onnx-tensorrt parser (официальный парсер NVIDIA)
      3. torch_tensorrt JIT (последний резерв, работает для DeepLab/FCN)

    Если все стратегии не удались — создаётся OnnxTrtFallbackSegmenter
    (ONNX Runtime + CUDAExecutionProvider) как высокопроизводительный fallback.

    Args:
        model: PyTorch модель (nn.Module) в eval режиме.
        model_key: Ключ модели для именования файлов (напр. "unet_smp_none").
        output_dir: Директория для сохранения артефактов.
        input_shape: (B, C, H, W) — форма входа.
        opset_version: ONNX opset (рекомендуется ≥17).
        trt_precision: "fp32", "bf16" или "fp16".
        device: "cuda" или "cpu".
        dynamic_axes: Использовать динамические оси в ONNX.

    Returns:
        Dict с путями:
          - "onnx": Path к .onnx или None
          - "trt": Path к .trt или None
          - "trt_strategy": строка с использованной стратегией
          - "trt_fallback": OnnxTrtFallbackSegmenter или None
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Optional[Union[Path, Any]]] = {
        "onnx": None,
        "trt": None,
        "trt_strategy": None,
        "trt_fallback": None,
    }

    # ── Шаг 1: Экспорт в ONNX ────────────────────────────────────────
    onnx_path: Path = output_dir / f"{model_key}.onnx"

    # Переводим модель на CPU для стабильного ONNX экспорта
    model_cpu: nn.Module = model.to("cpu").eval()
    _freeze_model_for_export(model_cpu)

    dummy_input: torch.Tensor = torch.randn(*input_shape, dtype=torch.float32)

    try:
        dyn_axes: Optional[Dict[str, Dict[int, str]]] = None
        if dynamic_axes:
            dyn_axes: Dict[str, Dict[int, str]] = {
                "input": {0: "batch", 2: "height", 3: "width"},
                "output": {0: "batch", 2: "height", 3: "width"},
            }

        with torch.no_grad():
            torch.onnx.export(
                model_cpu,
                (dummy_input,),
                str(onnx_path),
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes=dyn_axes,
                verbose=False,
            )

        # Валидация + упрощение
        import onnx

        onnx_model: Any = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)

        try:
            from onnxsim import simplify

            ok: bool
            simplified, ok = simplify(onnx_model)
            if ok:
                onnx.save(simplified, str(onnx_path))
                logger.info("✅ ONNX simplified")
        except Exception as e:
            logger.warning(f"⚠️ ONNX simplifier failed: {e}")
            pass

        results["onnx"] = onnx_path
        size_mb: float = onnx_path.stat().st_size / 1e6
        logger.info(f"✅ ONNX exported: {onnx_path} ({size_mb:.1f} MB)")

    except Exception as e:
        logger.error(f"❌ ONNX export failed для {model_key}: {e}")
        if device == "cuda" and torch.cuda.is_available():
            model.to(device)
        return results

    finally:
        # Возвращаем модель на CUDA
        del model_cpu
        gc.collect()
        if device == "cuda" and torch.cuda.is_available():
            model.to(device)

    # ── Шаг 2: ONNX → TRT ────────────────────────────────────────────
    if device != "cuda" or not torch.cuda.is_available():
        logger.info("TRT экспорт пропущен: не CUDA")
        return results

    trt_path = output_dir / f"{model_key}.{trt_precision}.trt"

    logger.info(f"🔍 TRT build checklist: {CHECKLIST}")
    if not CHECKLIST["tensorrt_api"]:
        logger.warning("⚠️ tensorrt API недоступен, пробуем fallback-стратегии")

    # Стратегия 1: tensorrt Python API
    if CHECKLIST["tensorrt_api"]:
        logger.info(f"[TRT Strategy 1] tensorrt API: {model_key}")
        success: bool = export_onnx_to_trt_via_api(
            onnx_path=onnx_path,
            trt_path=trt_path,
            precision=trt_precision,
            input_shape=input_shape,
            workspace_gb=4.0,
        )
        if success:
            results["trt"] = trt_path
            results["trt_strategy"] = "tensorrt_api"
            logger.info(f"✅ TRT via tensorrt API: {trt_path}")
            return results

    # Стратегия 2: trtexec
    logger.info(f"[TRT Strategy 2] trtexec: {model_key}")
    success = export_onnx_to_trt_via_trtexec(
        onnx_path=onnx_path,
        trt_path=trt_path,
        precision=trt_precision,
        input_shape=input_shape,
    )
    if success:
        results["trt"] = trt_path
        results["trt_strategy"] = "trtexec"
        logger.info(f"✅ TRT via trtexec: {trt_path}")
        return results

    # Стратегия 2.5: onnx-tensorrt parser (новая)
    if _check_import("onnx_tensorrt"):
        logger.info(f"[TRT Strategy 2.5] onnx-tensorrt: {model_key}")
        success = export_onnx_to_trt_via_onnx_tensorrt(
            onnx_path=onnx_path,
            trt_path=trt_path,
            precision=trt_precision,
            input_shape=input_shape,
        )
        if success:
            results["trt"] = trt_path
            results["trt_strategy"] = "onnx_tensorrt"
            logger.info(f"✅ TRT via onnx-tensorrt: {trt_path}")
            return results

    # Стратегия 3: torch_tensorrt JIT (работает для DeepLab/FCN)
    if CHECKLIST["torch_tensorrt"]:
        logger.info(f"[TRT Strategy 3] torch_tensorrt JIT: {model_key}")
        success = _export_via_torch_tensorrt_jit(
            model=model,
            trt_path=trt_path,
            precision=trt_precision,
            input_shape=input_shape,
            device=device,
        )
        if success:
            results["trt"] = trt_path
            results["trt_strategy"] = "torch_tensorrt_jit"
            logger.info(f"✅ TRT via torch_tensorrt JIT: {trt_path}")
            return results

    # Fallback: ONNX Runtime с CUDA provider
    logger.warning(
        f"⚠️ TRT не удался для {model_key}. "
        f"Используем ONNX Runtime + CUDAExecutionProvider как высокопроизводительный fallback."
    )
    logger.info(f"🔍 {_diagnose_onnx_cuda()}")
    try:
        fallback: OnnxTrtFallbackSegmenter = OnnxTrtFallbackSegmenter(
            onnx_path=onnx_path,
            input_shape=input_shape,
            device=device,
        )
        results["trt_fallback"] = fallback
        results["trt_strategy"] = "onnx_cuda_fallback"
        logger.info(f"✅ ONNX+CUDA fallback готов для {model_key}")
    except Exception as e:
        logger.error(f"❌ ONNX fallback тоже не удался: {e}")

    return results


def _freeze_model_for_export(model: nn.Module) -> None:
    """Переводит модель в eval и убирает training-only поведение."""
    model.eval()
    # Фиксируем BatchNorm статистику
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm)):
            m.training = False
    # Убираем Dropout
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.training = False


def _export_via_torch_tensorrt_jit(
    model: nn.Module,
    trt_path: Path,
    precision: str,
    input_shape: Tuple[int, int, int, int],
    device: str,
) -> bool:
    """Fallback: экспорт через torch_tensorrt.compile (JIT)."""
    try:
        import torch_tensorrt as torchtrt
    except ImportError:
        logger.warning("torch_tensorrt не установлен")
        return False

    try:
        model_cuda: nn.Module = model.to(device).eval()
        dummy: torch.Tensor = torch.randn(*input_shape, device=device, dtype=torch.float32)

        # trace
        traced: torch.jit.ScriptModule = torch.jit.trace(model_cuda, dummy, check_trace=False)

        enabled_precisions: Set[torch.dtype] = {torch.float32}
        if precision == "fp16" and torch.cuda.get_device_capability()[0] >= 6:
            enabled_precisions.add(torch.float16)
        if precision == "bf16":
            if torch.cuda.get_device_capability()[0] >= 6:
                logger.warning("⚠️ torch_tensorrt не поддерживает bf16, используем fp16 как fallback")
                enabled_precisions.add(torch.float16)
            else:
                logger.warning("⚠️ torch_tensorrt: GPU не поддерживает fp16, используем fp32")

        ir_mode: Literal["dynamo", "ts"] = _get_trt_ir_mode()
        logger.info(f"Using torch_tensorrt IR mode: {ir_mode}")

        trt_model: Any = torchtrt.compile(
            traced,
            inputs=[
                torchtrt.Input(
                    min_shape=input_shape,
                    opt_shape=input_shape,
                    max_shape=input_shape,
                    dtype=torch.float32,
                )
            ],
            enabled_precisions=enabled_precisions,
            ir=ir_mode,
            require_full_compilation=True,
            truncate_long_and_double=True,
        )

        torch.jit.save(trt_model, str(trt_path))
        logger.info(f"✅ TRT via torch_tensorrt JIT: {trt_path}")
        return True

    except Exception as e:
        logger.error(f"torch_tensorrt JIT failed: {e}")
        if trt_path.exists():
            trt_path.unlink()
        if os.path.exists(trt_path):
            os.remove(trt_path)
        return False


# ─────────────────────────────────────────────────────────────────────
# Загрузка TRT engine (поддерживает все форматы)
# ─────────────────────────────────────────────────────────────────────
def load_trt_engine(
    trt_path: Union[str, Path],
    device: str = "cuda",
) -> Optional[Any]:
    """Загружает TRT engine.

    Поддерживает все три формата:
      - tensorrt serialized engine (.trt от tensorrt API / trtexec)
      - torch.jit TorchScript (.trt от torch_tensorrt JIT)
      - torch_tensorrt dynamo

    Returns:
        Callable модель или None.
    """
    trt_path = Path(trt_path)
    if not trt_path.exists():
        logger.error(f"TRT файл не найден: {trt_path}")
        return None

    # Формат 1: tensorrt API / trtexec serialized engine
    # Узнаём по первым байтам: TRT engines начинаются с "ptrt" или специального magic
    try:
        import tensorrt as trt

        TRT_LOGGER: trt.Logger = trt.Logger(trt.Logger.WARNING)
        runtime: trt.Runtime = trt.Runtime(TRT_LOGGER)

        with open(trt_path, "rb") as f:
            engine_bytes: bytes = f.read()

        engine: Optional[trt.ICudaEngine]
        if hasattr(runtime, "deserialize_cuda_engine"):
            engine = runtime.deserialize_cuda_engine(engine_bytes)
        elif hasattr(runtime, "deserialize_engine"):  # TRT >= 8.5
            engine = runtime.deserialize_engine(engine_bytes)
        else:
            logger.error("❌ Неизвестная версия TensorRT API")
            return None
        if engine is not None:
            logger.info(f"✅ Loaded via tensorrt Runtime: {trt_path}")
            return TrtEngineWrapper(engine, device)
    except Exception as e:
        logger.warning(f"⚠️ TRT Runtime load failed: {e}")
        pass

    # Формат 2: torch.jit (torch_tensorrt JIT) TorchScript
    try:
        model: nn.Module = torch.jit.load(str(trt_path))
        model = model.to(device).eval()
        logger.info(f"✅ Loaded via torch.jit.load: {trt_path}")
        return model
    except Exception as e:
        logger.warning(f"⚠️ torch_tensorrt JIT aka TorchScript load failed: {e}")
        pass

    # Формат 3: torch_tensorrt dynamo
    try:
        import torch_tensorrt as torchtrt

        model = torchtrt.load(str(trt_path))
        logger.info(f"✅ Loaded via torch_tensorrt.load: {trt_path}")
        return model
    except Exception as e:
        logger.error(f"❌ Все методы загрузки TRT не удались для {trt_path}: {e}")
        return None


class TrtEngineWrapper:
    """Обёртка для tensorrt.ICudaEngine, предоставляющая интерфейс nn.Module.

    Позволяет использовать TRT engines, собранные через tensorrt API или trtexec,
    так же как обычные PyTorch модели.
    """

    def __init__(self, engine: Any, device: str = "cuda") -> None:
        """Инициализация модуля TrtEngineWrapper."""
        import tensorrt as trt

        self.engine: trt.ICudaEngine = engine
        self.device: str = device
        self.context: trt.IExecutionContext = engine.create_execution_context()

        # Выясняем имена и индексы входов/выходов
        self.input_name: Optional[str] = None
        self.output_name: Optional[str] = None
        for i in range(engine.num_io_tensors):
            name: str = engine.get_tensor_name(i)
            mode: trt.TensorIOMode = engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_name = name

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Инференс через TRT context."""
        x = x.contiguous().to(self.device).float()
        b, c, h, w = x.shape

        # Устанавливаем форму входа (для dynamic shapes)
        self.context.set_input_shape(self.input_name, (b, c, h, w))

        # Выходной буфер
        out_shape: Tuple[int, ...] = tuple(self.context.get_tensor_shape(self.output_name))
        output: torch.Tensor = torch.empty(out_shape, dtype=torch.float32, device=self.device)

        # Выполняем инференс
        self.context.set_tensor_address(self.input_name, x.data_ptr())
        self.context.set_tensor_address(self.output_name, output.data_ptr())

        stream: int = torch.cuda.current_stream().cuda_stream
        if hasattr(self.context, "execute_async_v3"):
            self.context.execute_async_v3(stream)
        elif hasattr(self.context, "execute_async_v2"):
            self.context.execute_async_v2(stream)
        else:
            # Fallback на синхронное выполнение
            self.context.execute()
        torch.cuda.synchronize()

        return output

    def eval(self) -> "TrtEngineWrapper":
        """Возвращает созданный TrtEngineWrapper."""
        return self


# ─────────────────────────────────────────────────────────────────────
# Сохранение совместимости со старым load_trt_model
# ─────────────────────────────────────────────────────────────────────
def load_trt_model(path: Union[str, Path], **kwargs: Any) -> Optional[Any]:
    """Backward-compatible alias для load_trt_engine."""
    return load_trt_engine(path, **kwargs)
