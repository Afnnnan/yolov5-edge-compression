"""
COCO dataset utilities for the model compression pipeline.

Provides:
- COCO val2017 download
- Image loading with YOLOv5 preprocessing
- Calibration subset creation
- Detection format conversion (YOLO → COCO JSON)
"""

import os
import json
import random
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
from tqdm import tqdm

from utils.common import setup_logger, COCO_CLASS_IDS

logger = setup_logger("coco_utils")


# =============================================================================
# Dataset Download
# =============================================================================

def download_coco_val2017(data_dir: str = "data/coco"):
    """
    Download COCO val2017 images and annotations.
    Skips download if files already exist.
    """
    data_dir = Path(data_dir)
    images_dir = data_dir / "val2017"
    ann_file = data_dir / "annotations" / "instances_val2017.json"

    if images_dir.exists() and ann_file.exists():
        num_images = len(list(images_dir.glob("*.jpg")))
        if num_images >= 5000:
            logger.info(f"COCO val2017 already downloaded ({num_images} images)")
            return str(images_dir), str(ann_file)

    os.makedirs(data_dir, exist_ok=True)

    # Download images
    if not images_dir.exists() or len(list(images_dir.glob("*.jpg"))) < 5000:
        logger.info("Downloading COCO val2017 images (~1GB)...")
        img_url = "http://images.cocodataset.org/zips/val2017.zip"
        img_zip = data_dir / "val2017.zip"
        subprocess.run(["wget", "-q", "--show-progress", "-O", str(img_zip), img_url], check=True)
        subprocess.run(["unzip", "-q", "-o", str(img_zip), "-d", str(data_dir)], check=True)
        os.remove(img_zip)
        logger.info("Images downloaded and extracted.")

    # Download annotations
    if not ann_file.exists():
        logger.info("Downloading COCO annotations...")
        ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        ann_zip = data_dir / "annotations.zip"
        subprocess.run(["wget", "-q", "--show-progress", "-O", str(ann_zip), ann_url], check=True)
        subprocess.run(["unzip", "-q", "-o", str(ann_zip), "-d", str(data_dir)], check=True)
        os.remove(ann_zip)
        logger.info("Annotations downloaded and extracted.")

    return str(images_dir), str(ann_file)


# =============================================================================
# Image Preprocessing (YOLOv5 letterbox)
# =============================================================================

def letterbox(
    image: np.ndarray,
    new_shape: int = 640,
    color: Tuple[int, int, int] = (114, 114, 114),
    auto: bool = False,
    scaleFill: bool = False,
    scaleup: bool = True,
    stride: int = 32,
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """
    Resize and pad image to target size while maintaining aspect ratio.
    This is the standard YOLOv5 letterbox preprocessing.

    Returns:
        resized_image: Letterboxed image
        ratio: Scale ratio (new / old)
        padding: (dw, dh) padding applied
    """
    shape = image.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    # Compute new unpadded dimensions
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]

    if auto:
        dw = dw % stride
        dh = dh % stride
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        r = new_shape[1] / shape[1]

    # Divide padding evenly
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return image, r, (dw, dh)


def preprocess_image(
    image_path: str,
    input_size: int = 640,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Load and preprocess a single image for YOLOv5 inference.

    Returns:
        input_tensor: Preprocessed tensor (1, 3, H, W) float32 [0, 1]
        original_image: Original BGR image
        meta: Dict with preprocessing metadata for postprocessing
    """
    # Load image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Failed to load image: {image_path}")

    orig_h, orig_w = img_bgr.shape[:2]

    # Letterbox resize
    img_resized, ratio, (dw, dh) = letterbox(img_bgr, new_shape=input_size)

    # BGR → RGB, HWC → CHW, normalize to [0, 1]
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_chw = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    input_tensor = np.expand_dims(img_chw, axis=0)  # Add batch dim
    input_tensor = np.ascontiguousarray(input_tensor)

    meta = {
        "orig_shape": (orig_h, orig_w),
        "ratio": ratio,
        "pad": (dw, dh),
        "input_size": input_size,
    }

    return input_tensor, img_bgr, meta


# =============================================================================
# Calibration Dataset
# =============================================================================

class CalibrationDataLoader:
    """
    Iterator over calibration images for TensorRT INT8 calibration.
    Loads a random subset of COCO val2017 images with YOLOv5 preprocessing.
    """

    def __init__(
        self,
        images_dir: str,
        num_images: int = 500,
        input_size: int = 640,
        batch_size: int = 1,
        seed: int = 42,
    ):
        self.images_dir = Path(images_dir)
        self.input_size = input_size
        self.batch_size = batch_size

        # Get all image paths and sample a subset
        all_images = sorted(self.images_dir.glob("*.jpg"))
        if len(all_images) == 0:
            raise FileNotFoundError(f"No images found in {images_dir}")

        random.seed(seed)
        self.image_paths = random.sample(all_images, min(num_images, len(all_images)))
        self.num_images = len(self.image_paths)
        self.index = 0

        # Calculate byte size for one batch
        self.input_nbytes = batch_size * 3 * input_size * input_size * np.dtype(np.float32).itemsize

        logger.info(
            f"Calibration dataset: {self.num_images} images, "
            f"batch_size={batch_size}, input_size={input_size}"
        )

    def __len__(self):
        return self.num_images // self.batch_size

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self) -> np.ndarray:
        if self.index >= self.num_images:
            raise StopIteration

        batch_images = []
        for _ in range(self.batch_size):
            if self.index >= self.num_images:
                break
            img_path = str(self.image_paths[self.index])
            input_tensor, _, _ = preprocess_image(img_path, self.input_size)
            batch_images.append(input_tensor)
            self.index += 1

        batch = np.concatenate(batch_images, axis=0)
        return np.ascontiguousarray(batch, dtype=np.float32)

    def next_batch(self) -> Optional[np.ndarray]:
        """Convenience method for the TensorRT calibrator interface."""
        try:
            return self.__next__()
        except StopIteration:
            return None


# =============================================================================
# Postprocessing — NMS & Format Conversion
# =============================================================================

def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    """Convert [x_center, y_center, w, h] to [x1, y1, x2, y2]."""
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # x1
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # y1
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # x2
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # y2
    return y


def numpy_nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Non-Maximum Suppression implemented in NumPy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h
        iou = intersection / (areas[i] + areas[order[1:]] - intersection + 1e-6)

        mask = iou <= iou_threshold
        order = order[1:][mask]

    return np.array(keep)


def postprocess_yolov5(
    raw_output: np.ndarray,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_detections: int = 300,
) -> List[np.ndarray]:
    """
    Postprocess raw YOLOv5 output tensor.

    Supports two output formats:
      - YOLOv5 classic (85 channels): [x, y, w, h, obj_conf, cls0..cls79]
      - YOLOv5nu / YOLOv8-style (84 channels): [x, y, w, h, cls0..cls79]

    Args:
        raw_output: Raw model output, shape (batch, num_anchors, 84 or 85)
        conf_threshold: Minimum confidence threshold
        iou_threshold: NMS IoU threshold
        max_detections: Maximum detections per image

    Returns:
        List of detection arrays per batch image, each (N, 6):
        [x1, y1, x2, y2, confidence, class_id]
    """
    batch_size = raw_output.shape[0]
    num_channels = raw_output.shape[2]
    results = []

    # Auto-detect output format
    # 84 channels = YOLOv5nu (no objectness): [x,y,w,h, 80 classes]
    # 85 channels = classic YOLOv5 (with objectness): [x,y,w,h, obj, 80 classes]
    has_objectness = (num_channels == 85)

    for i in range(batch_size):
        predictions = raw_output[i]  # (num_anchors, 84 or 85)

        if has_objectness:
            # Classic YOLOv5: confidence = obj_conf * class_prob
            obj_conf = predictions[:, 4]
            class_scores = predictions[:, 5:] * obj_conf[:, None]
        else:
            # YOLOv5nu / YOLOv8: confidence = class_prob directly
            class_scores = predictions[:, 4:]

        class_ids = class_scores.argmax(axis=1)
        class_confs = class_scores.max(axis=1)

        # Filter by confidence
        mask = class_confs > conf_threshold
        predictions = predictions[mask]
        class_ids = class_ids[mask]
        class_confs = class_confs[mask]

        if len(predictions) == 0:
            results.append(np.zeros((0, 6)))
            continue

        # Convert boxes from xywh to xyxy
        boxes = xywh2xyxy(predictions[:, :4])

        # Per-class NMS
        detections = []
        unique_classes = np.unique(class_ids)
        for cls in unique_classes:
            cls_mask = class_ids == cls
            cls_boxes = boxes[cls_mask]
            cls_scores_nms = class_confs[cls_mask]

            keep = numpy_nms(cls_boxes, cls_scores_nms, iou_threshold)
            for k in keep:
                detections.append([
                    cls_boxes[k, 0], cls_boxes[k, 1],
                    cls_boxes[k, 2], cls_boxes[k, 3],
                    cls_scores_nms[k], cls,
                ])

        if len(detections) == 0:
            results.append(np.zeros((0, 6)))
            continue

        detections = np.array(detections)
        # Sort by confidence and keep top-k
        order = detections[:, 4].argsort()[::-1][:max_detections]
        results.append(detections[order])

    return results


def scale_boxes_to_original(
    detections: np.ndarray,
    meta: dict,
) -> np.ndarray:
    """
    Scale detection boxes from model input space back to original image space.
    Reverses the letterbox transformation.
    """
    if len(detections) == 0:
        return detections

    dets = detections.copy()
    ratio = meta["ratio"]
    dw, dh = meta["pad"]
    orig_h, orig_w = meta["orig_shape"]

    # Remove padding
    dets[:, 0] -= dw
    dets[:, 1] -= dh
    dets[:, 2] -= dw
    dets[:, 3] -= dh

    # Remove scaling
    dets[:, :4] /= ratio

    # Clip to image bounds
    dets[:, 0] = np.clip(dets[:, 0], 0, orig_w)
    dets[:, 1] = np.clip(dets[:, 1], 0, orig_h)
    dets[:, 2] = np.clip(dets[:, 2], 0, orig_w)
    dets[:, 3] = np.clip(dets[:, 3], 0, orig_h)

    return dets


def detections_to_coco_json(
    detections: np.ndarray,
    image_id: int,
) -> List[dict]:
    """
    Convert detection array to COCO JSON format for pycocotools evaluation.

    Args:
        detections: (N, 6) array [x1, y1, x2, y2, confidence, class_id]
        image_id: COCO image ID

    Returns:
        List of COCO detection dicts
    """
    coco_results = []
    for det in detections:
        x1, y1, x2, y2, score, cls_id = det
        w = x2 - x1
        h = y2 - y1
        coco_results.append({
            "image_id": int(image_id),
            "category_id": COCO_CLASS_IDS[int(cls_id)],
            "bbox": [round(float(x1), 2), round(float(y1), 2),
                     round(float(w), 2), round(float(h), 2)],
            "score": round(float(score), 5),
        })
    return coco_results


def get_coco_image_ids(annotation_file: str) -> List[int]:
    """Get all image IDs from COCO annotation file."""
    with open(annotation_file, "r") as f:
        data = json.load(f)
    return [img["id"] for img in data["images"]]


def get_coco_image_path(images_dir: str, image_id: int) -> str:
    """Get image file path from COCO image ID."""
    return os.path.join(images_dir, f"{image_id:012d}.jpg")
