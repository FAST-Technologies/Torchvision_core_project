# utils/config_loader.py

from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml
import torch


def load_session_config(config_path: str = "configs/session.yaml") -> Dict[str, Any]:
    """Загружает и валидирует конфигурацию сессии."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Конфигурация не найдена: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    # Базовая валидация (можно расширить через pydantic/dataclasses)
    assert "session" in cfg, "Отсутствует секция 'session'"
    assert "flow" in cfg, "Отсутствует секция 'flow'"
    assert "test_settings" in cfg, "Отсутствует секция 'test_settings'"

    # Разрешаем пути относительно директории конфига
    base_dir = cfg["output"]["base_dir"]
    if not Path(base_dir).is_absolute():
        cfg["output"]["base_dir"] = str(Path(config_path).parent.parent / base_dir)

    # Нормализация вложенных ключей
    _normalize_nested_keys(cfg)

    # Авто-определение device/precision если указано "auto"
    _resolve_auto_settings(cfg)

    return cfg


def _normalize_nested_keys(cfg: Dict[str, Any], prefix: str = "") -> None:
    """Рекурсивная нормализация ключей (опционально)."""
    for key, value in list(cfg.items()):
        if isinstance(value, dict):
            _normalize_nested_keys(value, f"{prefix}.{key}" if prefix else key)


def _resolve_auto_settings(cfg: Dict[str, Any]) -> None:
    """Разрешает 'auto' значения для device и точности."""
    test_cfg = cfg.get("test_settings", {})

    # Device
    if test_cfg.get("device") == "auto":
        test_cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    # Precision
    if test_cfg.get("precision") == "auto":
        device = torch.device(test_cfg["device"])
        if device.type != "cuda":
            test_cfg["precision"] = "fp32"
        elif torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            if props.major >= 8:
                test_cfg["precision"] = "bf16"
            elif props.major >= 6:
                test_cfg["precision"] = "fp16"
            else:
                test_cfg["precision"] = "fp32"

    # torch.compile конфиг
    compile_cfg = test_cfg.setdefault("torch_compile", {})
    if compile_cfg.get("enabled") and compile_cfg.get("mode") is None:
        compile_cfg["mode"] = "reduce-overhead"
