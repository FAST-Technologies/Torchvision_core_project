import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env"}
    MODEL_DIR: str = "./models"
    DEFAULT_IMAGE: str = "./data/ade20k_test_trained/original_image_0.jpg"
    DEFAULT_GT: str = "./data/ade20k_test_trained/original_image_mask_0.png"

    # Пути к моделям
    SEGFORMER_PATH: str = "/home/yamshikov/models/segformer-b5-ready"
    UNET_CHECKPOINT: str = "unet_ade20k_best_200_epochs.pth"
    DEEPLAB_CHECKPOINT: str = "deeplab_ade20k_best_200_epochs.pth"
    FPN_MIT_CHECKPOINT: str = "fpn_mit_b5_ade20k_best_200_epochs.pth"
    PSP_MIT_CHECKPOINT: str = "psp_mit_b5_ade20k_best_200_epochs.pth"
    FCN_RESNET50_CHECKPOINT: str = "fcn_resnet50_ade20k_best_200_epochs.pth"
    SEGNET_RESNET34_CHECKPOINT: str = "segnet_ade20k_best_200_epochs.pth"


settings = Settings()
