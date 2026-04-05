# utils/strategies.py

# Импорт основных библиотек
import sys
import os

from .utils import extract_logits_info

import torchvision.transforms as T
import segmentation_models_pytorch as smp

from typing import (
    List,
    Union,
    Tuple,
    Dict,
    Any,
    Optional,
)
import torch
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import time
from scipy.ndimage import zoom

from utils.utils import (
    compute_metrics,
    analyze_prediction,
    generate_class_report,
    export_class_report,
)
from utils.palettes import ade_palette

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

num_classes: int = 150


def infer_segformer(
    model: Any, processor: Any, image: Image.Image, device: str = "cuda"
) -> Tuple[np.ndarray, Image.Image]:
    """Инференс для SegFormer — возвращает маску в размере оригинального изображения (работает для B0-B5)."""
    inputs = processor(images=image, return_tensors="pt").to(device)
    print(inputs)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    print(type(outputs[0]))
    logits_info = extract_logits_info(outputs, "segformer")
    print(f"📈 SegFormer logits: {logits_info}")
    print(f"📈 SegFormer custom logits: {logits}")
    seg_map = (
        processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]]  # [H, W] = [height, width]
        )[0]
        .cpu()
        .numpy()
    )
    print(seg_map)
    return seg_map, image


def infer_mask2former(
    model: Any, processor: Any, image: Image.Image, device: str = "cuda"
) -> Tuple[np.ndarray, Image.Image]:
    if image.width == 0 or image.height == 0:
        raise ValueError("Image has zero dimensions")

    inputs = processor(images=image, return_tensors="pt").to(device)
    print(inputs)
    with torch.no_grad():
        outputs = model(**inputs)
    print(type(outputs[0]))
    logits_info = extract_logits_info(outputs, "mask2former")
    print(f"📈 Mask2Former logits: {logits_info}")
    result = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image.size[::-1]]
    )[0]
    predicted_mask = result.cpu().numpy()
    print(predicted_mask)
    return predicted_mask, image


def infer_oneformer(
    model: Any, processor: Any, image: Image.Image, device: str = "cuda"
) -> Tuple[np.ndarray, Image.Image]:
    inputs = processor(images=image, task_inputs=["semantic"], return_tensors="pt").to(
        device
    )
    print(inputs)
    with torch.no_grad():
        outputs = model(**inputs)
    print(type(outputs[0]))
    logits_info = extract_logits_info(outputs, "oneformer")
    print(f"📈 OneFormer logits: {logits_info}")
    predicted_mask = (
        processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0]
        .cpu()
        .numpy()
    )
    print(predicted_mask)
    return predicted_mask, image


def infer_deeplab_torchvision(
    model: Any,
    processor: Any,
    image: Image.Image,
    device: str = "cuda",
    target_size: Tuple[int, int] = (512, 512),
) -> Tuple[np.ndarray, Image.Image]:
    """Инференс для DeepLabV3+ из torchvision"""

    # Preprocessing
    preprocess = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    print(preprocess)

    # Ресайз к target_size (как при обучении!)
    image_resized = image.resize(target_size, Image.BILINEAR)
    input_tensor = preprocess(image_resized).unsqueeze(0).to(device)
    print(input_tensor)

    with torch.no_grad():
        raw_output = model(input_tensor)

    logits_info = extract_logits_info(raw_output, "deeplab_tv")
    print(f"📈 DeepLabV3+ logits: {logits_info}")

    # Извлечение логитов
    if isinstance(raw_output, dict):
        logits = raw_output["out"][0]
    else:
        logits = (
            raw_output[0] if hasattr(raw_output, "__getitem__") else raw_output
        )  # [C, H, W]

    predicted_mask = logits.argmax(0).cpu().numpy()  # [H, W]

    # Ресайз к оригиналу
    if predicted_mask.shape != (image.size[1], image.size[0]):
        sh, sw = (
            image.size[1] / predicted_mask.shape[0],
            image.size[0] / predicted_mask.shape[1],
        )
        predicted_mask = zoom(predicted_mask, (sh, sw), order=0)
    print(predicted_mask)
    return predicted_mask, image


def infer_unet_smp(
    model: Any,
    processor: Any,
    image: Image.Image,
    encoder_name: str = "resnet34",
    device: str = "cuda",
    output_stride: int = 1,
) -> Tuple[np.ndarray, Image.Image]:
    """
    Универсальный инференс для SMP-моделей и SegNet.
    """
    # Preprocessing
    try:
        # Для SMP-моделей с encoder
        if hasattr(model, "encoder") and hasattr(model.encoder, "name"):
            encoder_name = model.encoder.name
            preprocess_fn = smp.encoders.get_preprocessing_fn(encoder_name, "imagenet")
        else:
            # Fallback для SegNet без encoder атрибута
            raise AttributeError("No encoder attribute")
    except Exception:
        # Стандартный ImageNet preprocessing для SegNet
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        def preprocess_fn(x):
            return (x.astype(np.float32) / 255.0 - mean) / std

    image_np = np.array(image)

    # Preprocessing
    input_tensor = preprocess_fn(image_np)
    print(input_tensor)
    input_tensor = (
        torch.from_numpy(input_tensor).permute(2, 0, 1).float().unsqueeze(0).to(device)
    )
    print(input_tensor)

    # Паддинг под output_stride
    pad_h, pad_w = 0, 0
    if output_stride > 1:
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        pad_h = (output_stride - h % output_stride) % output_stride
        pad_w = (output_stride - w % output_stride) % output_stride
        if pad_h > 0 or pad_w > 0:
            input_tensor = torch.nn.functional.pad(
                input_tensor, (0, pad_w, 0, pad_h), mode="reflect"
            )

    with torch.no_grad():
        outputs = model(input_tensor)  # [B, C, H, W]
    print(type(outputs[0]))
    is_segnet = "SegNet" in str(type(model))
    model_type = "segnet" if is_segnet else "unet_smp"
    logits_info = extract_logits_info(outputs, model_type)
    print(f"📈 {'SegNet' if is_segnet else 'SMP'} logits: {logits_info}")

    # Пост-процессинг: argmax + ресайз к оригиналу
    predicted_mask = outputs.argmax(1).squeeze(0).cpu().numpy()  # [H, W]

    # Кроппинг после паддинга
    if output_stride > 1 and (pad_h > 0 or pad_w > 0):
        predicted_mask = predicted_mask[: image.size[1], : image.size[0]]

    if predicted_mask.shape != image.size[::-1]:
        sh, sw = (
            image.size[1] / predicted_mask.shape[0],
            image.size[0] / predicted_mask.shape[1],
        )
        predicted_mask = zoom(predicted_mask, (sh, sw), order=0)
    print(predicted_mask)
    return predicted_mask, image


def infer_sam(
    model: Any, processor: Any, image: Image.Image, device: str = "cuda"
) -> Tuple[np.ndarray, Image.Image]:
    img_w, img_h = image.size

    # Инференс (без prompts=None!)
    results = model(image)
    print(results)

    print("📈 MobileSAM: instance segmentation (no class logits)")
    if results[0].masks is not None:
        print(
            f"   Masks: {results[0].masks.data.shape}, conf: {results[0].boxes.conf if hasattr(results[0], 'boxes') else 'N/A'}"
        )
    # Создаём семантическую карту из инстанс-масок
    seg_map = np.zeros((img_h, img_w), dtype=np.uint8)

    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        for i, mask in enumerate(masks, start=1):
            # mask имеет форму (H_mask, W_mask) — обычно меньше оригинала
            mask_bin = (mask > 0.5).astype(np.uint8)
            mask_pil = Image.fromarray(mask_bin)
            # Ресайз: image.size = (width, height) — именно так ждёт resize()
            mask_resized = np.array(mask_pil.resize((img_w, img_h), Image.NEAREST))
            # Заполняем только пустые пиксели (избегаем перекрытия)
            empty = seg_map == 0
            # Теперь размеры совпадают: seg_map[H,W] и mask_resized[H,W]
            seg_map[empty & (mask_resized > 0)] = i
    print(seg_map)  # [H, W], dtype=uint8
    return seg_map, image


def infer_dpt(
    model: Any, processor: Any, image: Image.Image, device: str = "cuda"
) -> Tuple[np.ndarray, Image.Image]:
    inputs = processor(images=image, return_tensors="pt").to(device)
    print(inputs)
    with torch.no_grad():
        outputs = model(**inputs)
    print(type(outputs[0]))
    logits_info = extract_logits_info(outputs, "dpt")
    print(f"📈 DPT logits: {logits_info}")

    print(
        f"📈 DPT logits: {outputs.logits.shape if hasattr(outputs, 'logits') else 'N/A'}"
    )
    seg_map = (
        processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0]
        .cpu()
        .numpy()
    )
    return seg_map, image


def infer_smp_model(
    model: Any,
    processor: Any,
    image: Image.Image,
    device: str = "cuda",
    output_stride: int = 32,
    log_logits: bool = True,
) -> Tuple[np.ndarray, Image.Image]:
    """
    Универсальный инференс для SMP-моделей (U-Net, FPN, PSPNet, DeepLabV3+)
    с авто-паддингом под output_stride.

    Args:
        output_stride: кратность размера (32 для FPN/DeepLab, 8 для PSPNet, 1 для U-Net)
    """
    orig_w, orig_h = image.size

    # Preprocessing
    try:
        encoder_name = model.encoder.name
    except Exception:
        encoder_name = "mit_b5"

    preprocess_fn = smp.encoders.get_preprocessing_fn(encoder_name, "imagenet")
    print(preprocess_fn)

    # Конвертация + препроцессинг
    image_np = np.array(image)
    input_tensor = preprocess_fn(image_np)
    input_tensor = torch.from_numpy(input_tensor).permute(2, 0, 1).float()

    # ПАДДИНГ под output_stride
    if output_stride > 1:
        h, w = input_tensor.shape[1], input_tensor.shape[2]
        pad_h = (output_stride - h % output_stride) % output_stride
        pad_w = (output_stride - w % output_stride) % output_stride

        if pad_h > 0 or pad_w > 0:
            input_tensor = torch.nn.functional.pad(
                input_tensor,
                (0, pad_w, 0, pad_h),  # left, right, top, bottom
                mode="reflect",  # или 'constant', value=0
            )

    # Добавляем batch dimension и переносим на device
    input_tensor = input_tensor.unsqueeze(0).to(device)
    print(input_tensor)

    # Инференс
    with torch.no_grad():
        outputs = model(input_tensor)  # [B, C, H_pad, W_pad]

    if log_logits:
        logits_info = extract_logits_info(outputs, "smp")
        print(f"📈 SMP logits: {logits_info}")

    # Пост-процессинг: argmax + кроппинг к оригиналу
    pred_mask = outputs.argmax(1).squeeze(0).cpu().numpy()  # [H_pad, W_pad]

    # КРОППИНГ к оригинальному размеру
    if output_stride > 1 and (pad_h > 0 or pad_w > 0):
        pred_mask = pred_mask[:orig_h, :orig_w]

    return pred_mask, image


def infer_smp_model_fixed(
    model: Any,
    image: Image.Image,
    device: str = "cuda",
    output_stride: int = 32,
    target_size: Tuple[int, int] = (512, 512),
) -> Tuple[np.ndarray, Image.Image]:
    """
    Инференс для SMP-моделей

    Args:
        target_size: Размер для ресайза (должен совпадать с обучением!)
        output_stride: 32 для FPN, 8 для PSPNet, 1 для U-Net
    """

    orig_w, orig_h = image.size

    # Ресайз к target_size
    image_resized = image.resize(target_size, Image.BILINEAR)

    # Получаем encoder_name и preprocessing функцию
    try:
        encoder_name = model.encoder.name
    except Exception:
        encoder_name = "resnet34"

    preprocess_fn = smp.encoders.get_preprocessing_fn(encoder_name, "imagenet")

    # Preprocessing
    image_np = np.array(image_resized)
    input_tensor = preprocess_fn(image_np)
    input_tensor = torch.from_numpy(input_tensor).permute(2, 0, 1).float()

    # Паддинг под output_stride
    h, w = input_tensor.shape[1], input_tensor.shape[2]
    pad_h = pad_w = 0
    if output_stride > 1:
        pad_h = (output_stride - h % output_stride) % output_stride
        pad_w = (output_stride - w % output_stride) % output_stride
        if pad_h > 0 or pad_w > 0:
            input_tensor = torch.nn.functional.pad(
                input_tensor, (0, pad_w, 0, pad_h), mode="reflect"
            )

    input_tensor = input_tensor.unsqueeze(0).to(device)

    # Инференс
    with torch.no_grad():
        outputs = model(input_tensor)

    logits_info = extract_logits_info(outputs, "smp")
    print(f"📈 SMP logits: {logits_info}")

    # Пост-процессинг
    pred_mask = outputs.argmax(1).squeeze(0).cpu().numpy()

    # Кроппинг паддинга
    if pad_h > 0 or pad_w > 0:
        pred_mask = pred_mask[: target_size[1], : target_size[0]]

    # Ресайз к оригинальному размеру
    if pred_mask.shape != (orig_h, orig_w):
        sh, sw = orig_h / pred_mask.shape[0], orig_w / pred_mask.shape[1]
        pred_mask = zoom(pred_mask, (sh, sw), order=0)

    return pred_mask, image


def infer_fcn_torchvision(
    model: Any, processor: Any, image: Image.Image, device: str = "cuda"
) -> Tuple[np.ndarray, Image.Image]:
    """Инференс для FCN из torchvision"""

    preprocess = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    input_tensor = preprocess(image).unsqueeze(0).to(device)
    print(input_tensor)

    with torch.no_grad():
        outputs = model(input_tensor)

    logits_info = extract_logits_info(outputs, "fcn_tv")
    print(f"📈 FCN logits: {logits_info}")

    if isinstance(outputs, dict):
        logits = outputs["out"][0]  # [C, H, W]
    elif isinstance(outputs, (tuple, list)):
        logits = outputs[0] if isinstance(outputs[0], torch.Tensor) else outputs[0][0]
    else:
        logits = outputs[0] if hasattr(outputs, "__getitem__") else outputs

    pred_mask = logits.argmax(0).cpu().numpy()

    # Ресайз к оригиналу
    if pred_mask.shape != image.size[::-1]:
        sh, sw = image.size[1] / pred_mask.shape[0], image.size[0] / pred_mask.shape[1]
        pred_mask = zoom(pred_mask, (sh, sw), order=0)

    return pred_mask, image


def infer_fcn_torchvision_fixed(
    model: Any,
    processor: Any,
    image: Image.Image,
    device: str = "cuda",
    target_size: Tuple[int, int] = (512, 512),
) -> Tuple[np.ndarray, Image.Image]:

    orig_w, orig_h = image.size

    # Ресайз к target_size
    image_resized = image.resize(target_size, Image.BILINEAR)

    # Preprocessing (ImageNet stats)
    preprocess = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    input_tensor = preprocess(image_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)

    if isinstance(outputs, dict):
        logits = outputs["out"][0]
    else:
        logits = outputs[0] if hasattr(outputs, "__getitem__") else outputs

    pred_mask = logits.argmax(0).cpu().numpy()

    # Ресайз к оригиналу
    if pred_mask.shape != (orig_h, orig_w):
        sh, sw = orig_h / pred_mask.shape[0], orig_w / pred_mask.shape[1]
        pred_mask = zoom(pred_mask, (sh, sw), order=0)

    return pred_mask, image


def infer_mask_rcnn(
    model: Any,
    processor: Any,
    image: Image.Image,
    device: str = "cuda",
    log_logits: bool = True,
    score_threshold: float = 0.5,
) -> Tuple[np.ndarray, Image.Image]:
    """
    Инференс Mask R-CNN с конверсией instance → semantic.
    """

    preprocess = T.Compose(
        [
            T.ToTensor(),
        ]
    )

    input_tensor = preprocess(image).unsqueeze(0).to(device)
    print(input_tensor)

    with torch.no_grad():
        outputs = model(input_tensor)

    logits_info = extract_logits_info(outputs, "maskrcnn_tv")
    print(f"📈 Mask R-CNN: {logits_info}")

    result = outputs[0]

    # Извлекаем маски и классы
    masks = result["masks"].cpu().numpy()  # [N, 1, H, W]
    labels = result["labels"].cpu().numpy()  # [N]
    scores = result["scores"].cpu().numpy()  # [N]

    # Фильтруем по confidence
    valid = scores > score_threshold
    masks = masks[valid]
    labels = labels[valid]

    print(f"📈 Mask R-CNN: {len(masks)} instances detected (score > {score_threshold})")
    print(f"   Detected {len(masks)} instances (score > {score_threshold})")
    if len(scores) > 0:
        print(f"   Score range: [{scores.min():.3f}, {scores.max():.3f}]")

    # Конвертация instance → semantic
    # Создаём семантическую карту, объединяя все маски
    img_h, img_w = image.size[1], image.size[0]
    semantic_map = np.zeros((img_h, img_w), dtype=np.uint8)
    for mask, label in zip(masks, labels):
        # mask: [1, H, W] → [H, W]
        mask_bin = (mask[0] > 0.5).astype(np.uint8)

        # Ресайз
        if mask_bin.shape != (img_h, img_w):
            mask_pil = Image.fromarray(mask_bin)
            mask_bin = np.array(mask_pil.resize((img_w, img_h), Image.NEAREST))

        semantic_map[mask_bin > 0] = label

    return semantic_map, image


class SegNet(torch.nn.Module):
    """
    Простая реализация SegNet для бенчмарка.
    Encoder-Decoder с max pooling indices.
    """

    def __init__(self, num_classes: int = num_classes) -> None:
        super().__init__()

        # Encoder (VGG16-like)
        self.enc1 = self._make_encoder(3, 64)
        self.enc2 = self._make_encoder(64, 128)
        self.enc3 = self._make_encoder(128, 256)
        self.enc4 = self._make_encoder(256, 512)
        self.enc5 = self._make_encoder(512, 512)

        # Decoder
        self.dec5 = self._make_decoder(512, 512)
        self.dec4 = self._make_decoder(512, 256)
        self.dec3 = self._make_decoder(256, 128)
        self.dec2 = self._make_decoder(128, 64)
        self.dec1 = self._make_decoder(64, 64)

        # Classifier
        self.classifier = torch.nn.Conv2d(64, num_classes, kernel_size=1)

    def _make_encoder(self, in_ch, out_ch):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, 3, padding=1),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_ch, out_ch, 3, padding=1),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True),
        )

    def _make_decoder(self, in_ch, out_ch):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, 3, padding=1),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_ch, out_ch, 3, padding=1),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_ch, out_ch, 3, padding=1),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        e1, p1 = self._encode(self.enc1, x)
        e2, p2 = self._encode(self.enc2, p1)
        e3, p3 = self._encode(self.enc3, p2)
        e4, p4 = self._encode(self.enc4, p3)
        e5, p5 = self._encode(self.enc5, p4)

        # Decoder
        d5 = self._decode(self.dec5, p5, e5.size())
        d4 = self._decode(self.dec4, d5, e4.size())
        d3 = self._decode(self.dec3, d4, e3.size())
        d2 = self._decode(self.dec2, d3, e2.size())
        d1 = self._decode(self.dec1, d2, e1.size())

        return self.classifier(d1)

    def _encode(self, encoder, x):
        """
        Encoder step: conv → batchnorm → relu → maxpool

        max_pool2d возвращает (output, indices), берём только output
        """
        x = encoder(x)
        pooled, indices = torch.nn.functional.max_pool2d(x, 2, 2, return_indices=True)
        # Возвращаем features для skip-connection и pooled для следующего слоя
        return x, pooled

    def _decode(self, decoder, x, output_size):
        """
        Decoder step: upsample → conv → batchnorm → relu
        """
        # Upsampling к размеру encoder features
        x = torch.nn.functional.interpolate(
            x, size=output_size[2:], mode="bilinear", align_corners=False
        )
        return decoder(x)


def segment_image_unified(
    model: Any,
    processor: Any,
    image_input: Union[str, Image.Image, np.ndarray],
    model_type: str,
    alpha: float = 0.5,
    palette: Optional[Union[List[List[int]], callable]] = None,
    device: str = "cuda",
    verbose: bool = True,
    num_classes: int = num_classes,
    class_names: Optional[dict] = None,
    gt_mask=None,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Универсальная функция сегментации для любой архитектуры.
    Используется в SegmentationBenchmark.

    Args:
        model: загруженная модель
        processor: процессор (для HF-моделей) или None
        image_input: путь к файлу, URL или PIL.Image
        model_type: "segformer" | "mask2former" | "deeplab_tv" | "unet_smp" | "sam"
        alpha: прозрачность маски (0..1)
        palette: палитра цветов (по умолчанию ADE20K)
        device: устройство для вычислений
        verbose: логировать детали
        num_classes: количество классов
        class_names: словарь имён классов
        gt_mask: ground truth для метрик

    Returns:
        Tuple[Image.Image, Dict]: (overlay, result_dict)
    """

    # Загрузка изображения
    if isinstance(image_input, str):
        if image_input.startswith(("http://", "https://")):
            resp = requests.get(image_input)
            image = Image.open(BytesIO(resp.content)).convert("RGB")
        else:
            image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    elif isinstance(image_input, np.ndarray):
        # Конвертация numpy array в PIL.Image
        if len(image_input.shape) == 2:
            # Grayscale → RGB
            image = Image.fromarray(image_input).convert("RGB")
        elif len(image_input.shape) == 3:
            if image_input.shape[2] == 3:
                # RGB
                image = Image.fromarray(image_input)
            elif image_input.shape[2] == 4:
                # RGBA → RGB
                image = Image.fromarray(image_input).convert("RGB")
            else:
                raise ValueError(
                    f"Unsupported number of channels: {image_input.shape[2]}"
                )
        else:
            raise ValueError(f"Unsupported array shape: {image_input.shape}")
    else:
        raise ValueError(
            f"Unsupported input type: {type(image_input)}. "
            f"Expected str, PIL.Image, or np.ndarray"
        )

    t0 = time.time()

    # Выбор стратегии инференса
    if model_type not in INFERENCE_STRATEGIES:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Available: {list(INFERENCE_STRATEGIES.keys())}"
        )

    infer_func = INFERENCE_STRATEGIES[model_type]

    # Выполнение инференса
    with torch.no_grad():
        seg_map, _ = infer_func(
            model=model, processor=processor, image=image, device=device
        )

    # Вербозный вывод и метрики
    result_info = {}
    overlay = _create_overlay_standalone(image, seg_map, alpha=alpha, palette=palette)
    if verbose:
        result_info = _log_inference_details_standalone(
            image=image,
            seg_map=seg_map,
            model_type=model_type,
            model=model,
            class_names=class_names,
            gt_mask=gt_mask,
            num_classes=num_classes,
            initial_time=t0,
            palette=palette,
        )
    else:
        # Минимальный result_info без verbose
        result_info = {
            "model": model_type,
            "overlay": overlay,
            "mask": seg_map,
            "inference_time_ms": (time.time() - t0) * 1000,
            "metrics": {},
            "image_size": image.size[::-1],
            "output_shape": seg_map.shape,
            "unique_classes": len(np.unique(seg_map)),
        }

    return overlay, result_info


def _log_inference_details_standalone(
    image: Image.Image,
    seg_map: np.ndarray,
    model_type: str,
    model: Any,
    class_names: Optional[dict] = None,
    gt_mask=None,
    num_classes: int = num_classes,
    initial_time: float = 0.0,
    palette=None,
) -> Dict[str, Any]:
    """Логирование деталей инференса (standalone версия)"""

    if callable(class_names):
        class_names = class_names()

    print(f"\n🔍 Model: {model_type}")
    print(f"   Mask shape: {seg_map.shape}, dtype: {seg_map.dtype}")

    unique_classes = np.unique(seg_map)
    print(
        f"   Predicted classes ({len(unique_classes)}): {unique_classes[:20]}{'...' if len(unique_classes) > 20 else ''}"
    )

    total_pixels = seg_map.size
    print("   Top 5 classes by pixel count:")
    class_stats = []
    for cls in unique_classes:
        count = np.sum(seg_map == cls)
        if count > 0:
            name = (
                class_names.get(cls, f"Class_{cls}") if class_names else f"Class_{cls}"
            )
            pct = 100 * count / total_pixels
            class_stats.append((cls, name, count, pct))
            print(f"     Class {cls:3d}: {count:6d} px ({pct:5.3f}%)")

    n_classes = _get_num_classes_standalone(model, model_type, fallback=num_classes)

    class_stats.sort(key=lambda x: x[2], reverse=True)
    for cls, name, count, pct in class_stats[:10]:
        print(f"     {cls:3d}: {name:25s} {count:7,} px ({pct:5.3f}%)")

    # Вывод информации о классах
    if hasattr(model, "config") and hasattr(model.config, "id2label"):
        print(f"\n   Class names (from model.config.id2label, {n_classes} total):")
        print("\n   ✅ HF model: class names from config.id2label")
    elif model_type in ["deeplab_tv", "fcn_tv", "maskrcnn_tv"]:
        print(f"\n   ⚠️  Torchvision models: {n_classes} output channels")
        print("      Class mapping depends on training dataset (COCO/ADE20K)")
    elif model_type in ["unet_smp", "mit_smp", "fpn_mit", "psp_mit", "segnet_custom"]:
        print(
            f"\n   ⚠️  SMP/Custom models: {n_classes} output channels (indices 0..{n_classes-1})"
        )
        print("      Mapping to ADE20K names requires external label file")
    elif model_type in ["sam", "mobile_sam", "sam2"]:
        print(
            f"\n   ⚠️  SAM models: instance IDs (1..{len(unique_classes)}), not semantic classes"
        )
        print("      Each detected object gets a unique ID, no class names")

    # Анализ предсказания
    try:
        analyze_prediction(seg_map, class_names=class_names)
        report = generate_class_report(seg_map, class_names=class_names)
        print(f"\n📊 Coverage: {report['coverage_pct']}% valid pixels")
        print(f"🏆 Top class: {report['top_class']} ({report['top_class_pct']}%)")
        export_class_report(
            report,
            f"./data/ade20k_test_trained/{model_type}_prediction_report.md",
            format="markdown",
        )
    except Exception as e:
        print(f"⚠️  Analysis skipped: {e}")

    # Метрики если есть GT
    metrics = {}
    if gt_mask is not None:
        try:
            gt_np = np.array(gt_mask) if isinstance(gt_mask, Image.Image) else gt_mask
            metrics = compute_metrics(seg_map, gt_np, num_classes=num_classes)
            print(f"   ✅ Metrics computed: IoU={metrics.get('iou', 0):.4f}")
        except Exception as e:
            print(f"⚠️  Metrics computation failed: {e}")
    else:
        print("⚠️  GT mask not provided, metrics skipped")

    # Создание overlay
    alpha = 0.5
    overlay = _create_overlay_standalone(image, seg_map, alpha=alpha, palette=palette)
    inference_time = time.time() - initial_time

    return {
        "model": model_type,
        "overlay": overlay,
        "mask": seg_map,
        "inference_time_ms": inference_time * 1000,
        "metrics": metrics,
        "image_size": image.size[::-1],
        "output_shape": seg_map.shape,
        "unique_classes": len(unique_classes),
        "class_stats": class_stats[:10],
    }


def _get_num_classes_standalone(
    model: Any, model_type: str, fallback: int = num_classes
) -> int:
    """Безопасно получает число классов из модели (standalone версия)"""
    try:
        if hasattr(model, "config") and hasattr(model.config, "id2label"):
            return len(model.config.id2label)

        if model_type in ["deeplab_tv", "fcn_tv"]:
            if hasattr(model, "classifier"):
                return model.classifier[-1].out_channels
            elif hasattr(model, "out_channels"):
                return model.out_channels

        if model_type in ["unet_smp", "mit_smp", "fpn_mit", "psp_mit", "segnet_custom"]:
            for module in reversed(list(model.modules())):
                if isinstance(module, torch.nn.Conv2d):
                    return module.out_channels
            return fallback

        if model_type in ["sam", "mobile_sam", "sam2"]:
            return None
        return fallback
    except Exception:
        return fallback


def _create_overlay_standalone(
    image: Image.Image, mask: np.ndarray, alpha: float = 0.5, palette=None
) -> Image.Image:
    """
    Создаёт визуализацию: оригинал + цветная маска (standalone версия).

    Args:
        image: PIL.Image в RGB
        mask: np.ndarray [H, W] с целочисленными метками классов
        alpha: коэффициент блендинга (0 = только фото, 1 = только маска)
        palette: список цветов [R,G,B] для каждого класса (по умолчанию ADE20K)

    Returns:
        PIL.Image с наложенной маской
    """

    if palette is None:
        palette = ade_palette()
    elif callable(palette):
        palette = palette()

    palette = np.array(palette, dtype=np.uint8)

    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for label, color in enumerate(palette[: mask.max() + 1]):
        color_mask[mask == label] = color

    img_arr = np.array(image.convert("RGB"))
    overlay = (img_arr * (1 - alpha) + color_mask * alpha).astype(np.uint8)

    return Image.fromarray(overlay)


INFERENCE_STRATEGIES = {
    # === Transformer-based HuggingFace модели ===
    "segformer": infer_segformer,
    "segformer_b2": infer_segformer,
    "mask2former": infer_mask2former,
    "maskformer": infer_mask2former,
    "oneformer": infer_oneformer,
    "dpt": infer_dpt,
    "upernet": infer_mask2former,
    # Torchvision модели
    "deeplab_tv": infer_deeplab_torchvision,
    "fcn_tv": infer_fcn_torchvision_fixed,
    "maskrcnn_tv": infer_mask_rcnn,
    # SMP модели с разными output_stride
    "unet_smp": lambda model, processor, image, device: infer_smp_model_fixed(
        model, image, device, output_stride=1
    ),
    "mit_smp": lambda model, processor, image, device: infer_smp_model_fixed(
        model, image, device, output_stride=1
    ),
    "fpn_mit": lambda model, processor, image, device: infer_smp_model_fixed(
        model, image, device, output_stride=32
    ),
    "fpn_effnet": lambda model, processor, image, device: infer_smp_model_fixed(
        model, image, device, output_stride=32
    ),
    "fpn_se_resnext": lambda model, processor, image, device: infer_smp_model_fixed(
        model, image, device, output_stride=32
    ),
    "psp_mit": lambda model, processor, image, device: infer_smp_model_fixed(
        model, image, device, output_stride=8
    ),
    "psp_effnet": lambda model, processor, image, device: infer_smp_model_fixed(
        model, image, device, output_stride=8
    ),
    "psp_resnet": lambda model, processor, image, device: infer_smp_model_fixed(
        model, image, device, output_stride=8
    ),
    "deeplab_smp": lambda model, processor, image, device: infer_smp_model_fixed(
        model, image, device, output_stride=16
    ),
    "segnet": lambda model, processor, image, device: infer_unet_smp(
        model, processor, image, device=device, output_stride=1
    ),
    "segnet_custom": lambda model, processor, image, device: infer_unet_smp(
        model, processor, image, device=device, output_stride=1
    ),
    # SAM семейство
    "sam": infer_sam,
    "mobile_sam": infer_sam,
    "sam2": infer_sam,
}
