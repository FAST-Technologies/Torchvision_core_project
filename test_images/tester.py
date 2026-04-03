# main.py

from transformers import MaskFormerImageProcessor, MaskFormerForInstanceSegmentation

from torch.utils.data import Dataset, DataLoader

from typing import (
    List,
    Union,
    Tuple,
    Dict,
    Any,
    TypeVar,
    Optional,
    Literal,
    Protocol,
    runtime_checkable,
    overload,
    TYPE_CHECKING,
)

import os
import sys
from datetime import datetime
import traceback
import warnings
import time
import requests
from io import BytesIO
from PIL import Image
import json

from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import gc

image_paths = {
    "war_frame_1": "2340_frame.jpg",
    "war_frame_2": "3330_frame.jpg",
    "war_frame_3": "4130_frame.jpg",
    "war_frame_4": "4480_frame.jpg",
    "building": "test_gt_image.jpg",
    "animals": "animals.jpg",
    "test_image_architecture": "test_image_architecture.jpg",
    "test_image_countryside": "test_image_countryside.jpg",
    "test_image_mountain": "test_image_mountain.jpg",
    "test_image_traffic": "test_image_traffic.jpg",
    "test_image_trucks": "test_image_trucks.jpg",
    "test_image_nature": "test_image_nature.jpg",
}

for name, path in image_paths.items():
    try:
        img = Image.open(path)
        print(f"✅ {name}: {img.size}, info: {img.info}")

    except Exception as e:
        print(f"❌ Ошибка загрузки {name}: {e}")
