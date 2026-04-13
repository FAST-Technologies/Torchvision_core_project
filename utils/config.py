import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydantic_settings import BaseSettings
from utils.paths import PROJECT_ROOT, MODELS_DIR, ADE20K_DIR


class Settings(BaseSettings):
    model_config = {"env_file": ".env"}
    MODEL_DIR: Path = MODELS_DIR
    DEFAULT_IMAGE: Path = ADE20K_DIR / "original_image_0.jpg"
    DEFAULT_GT: Path = ADE20K_DIR / "original_image_mask_0.png"

    # Пути к моделям
    SEGFORMER_PATH: str = "segformer-b5-ready"
    UNET_CHECKPOINT: str = "unet_ade20k_best_200_epochs.pth"
    DEEPLAB_CHECKPOINT: str = "deeplab_ade20k_best_200_epochs.pth"
    FPN_MIT_CHECKPOINT: str = "fpn_mit_b5_ade20k_best_200_epochs.pth"
    PSP_MIT_CHECKPOINT: str = "psp_mit_b5_ade20k_best_200_epochs.pth"
    FCN_RESNET50_CHECKPOINT: str = "fcn_resnet50_ade20k_best_200_epochs.pth"
    SEGNET_RESNET34_CHECKPOINT: str = "segnet_ade20k_best_200_epochs.pth"


settings = Settings()
