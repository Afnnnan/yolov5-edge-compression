"""
Script to run visual demo inference.
Compares PyTorch FP32 and TensorRT predictions side-by-side.
"""

import os
import sys
import random
import argparse
import time

import cv2
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.common import load_config, setup_logger, COCO_CLASS_NAMES
from utils.coco_utils import (
    get_coco_image_ids, get_coco_image_path, preprocess_image,
    postprocess_yolov5, scale_boxes_to_original
)
from ultralytics import YOLO

logger = setup_logger("demo_inference")


def get_color_palette(num_classes):
    """Generate distinct colors for each class."""
    np.random.seed(42)
    return [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(num_classes)]


def draw_detections(image, detections, colors):
    """Draw bounding boxes and labels on image."""
    img = image.copy()
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        color = colors[int(cls_id)]
        label = f"{COCO_CLASS_NAMES[int(cls_id)]} {conf:.2f}"

        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(img, label, (int(x1), max(0, int(y1) - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img


def run_pytorch(model, img_path):
    """Run inference with PyTorch YOLOv5n."""
    t0 = time.perf_counter()
    results = model(img_path, verbose=False)
    t1 = time.perf_counter()

    det = results[0].boxes.data.cpu().numpy()
    return det, (t1 - t0) * 1000


def run_trt(engine, img_path, conf_thres, iou_thres):
    """Run inference with TensorRT engine."""
    input_size = engine.get_input_shape()[-1]

    t0 = time.perf_counter()
    input_tensor, _, meta = preprocess_image(img_path, input_size)
    outputs = engine.infer(input_tensor)
    out = outputs[0]

    # Transpose (1, 84, 8400) -> (1, 8400, 84) if needed
    if out.shape[1] < out.shape[2]:
        out = np.transpose(out, (0, 2, 1))

    dets = postprocess_yolov5(out, conf_thres, iou_thres)[0]
    dets = scale_boxes_to_original(dets, meta)
    t1 = time.perf_counter()

    return dets, (t1 - t0) * 1000


def run_trt_debug(engine, img_path):
    """Run TRT inference and print raw output statistics for debugging."""
    input_size = engine.get_input_shape()[-1]
    input_tensor, _, meta = preprocess_image(img_path, input_size)
    outputs = engine.infer(input_tensor)
    out = outputs[0]

    print(f"\n  [DEBUG] Raw TRT output shape: {out.shape}")
    print(f"  [DEBUG] Raw output[0] min={out.min():.4f}, max={out.max():.4f}, mean={out.mean():.4f}")

    if out.shape[1] < out.shape[2]:
        out = np.transpose(out, (0, 2, 1))

    predictions = out[0]  # (8400, 84)
    box_coords = predictions[:, :4]
    class_scores = predictions[:, 4:]

    print(f"  [DEBUG] Box coords: min={box_coords.min():.2f}, max={box_coords.max():.2f}")
    print(f"  [DEBUG] Class scores: min={class_scores.min():.6f}, max={class_scores.max():.6f}, mean={class_scores.mean():.6f}")

    # Check how many detections pass various thresholds
    max_scores = class_scores.max(axis=1)
    for thresh in [0.001, 0.01, 0.1, 0.25, 0.5]:
        count = (max_scores > thresh).sum()
        print(f"  [DEBUG] Detections with conf > {thresh}: {count}")

    return class_scores


def main():
    parser = argparse.ArgumentParser(description="Run visual demo inference")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="Confidence threshold for visualization")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    args = parser.parse_args()

    config = load_config(args.config)
    demo_dir = config["paths"]["demo_dir"]
    os.makedirs(demo_dir, exist_ok=True)

    ann_file = config["paths"]["coco_annotations"]
    images_dir = config["paths"]["coco_images"]

    if not os.path.exists(ann_file) or not os.path.exists(images_dir):
        logger.error("COCO dataset not found. Please run evaluate_accuracy.py first to download.")
        sys.exit(1)

    image_ids = get_coco_image_ids(ann_file)
    random.seed(123)
    sample_ids = random.sample(image_ids, 8)

    colors = get_color_palette(len(COCO_CLASS_NAMES))

    # Load PyTorch model
    pt_model = YOLO(config["model"]["weights"])

    # Load TRT engine (try INT8 -> FP16 -> FP32)
    engine = None
    engine_name = "TRT INT8"
    for prec, key in [("INT8", "int8_engine"), ("FP16", "fp16_engine"), ("FP32", "fp32_engine")]:
        trt_path = config["paths"].get(key, f"outputs/yolov5n_{prec.lower()}.engine")
        if os.path.exists(trt_path):
            try:
                from utils.trt_inference import TRTInference
                engine = TRTInference(trt_path)
                engine_name = f"TRT {prec}"
                logger.info(f"Loaded {engine_name} engine from {trt_path}")
                break
            except Exception as e:
                logger.warning(f"Failed to load {prec} engine: {e}")

    conf_thres = args.conf_thres
    iou_thres = args.iou_thres

    # Debug: print raw output statistics for first image
    if engine:
        first_img = get_coco_image_path(images_dir, sample_ids[0])
        print("\n" + "=" * 60)
        print("DEBUG: TRT Raw Output Analysis")
        print("=" * 60)
        run_trt_debug(engine, first_img)
        print("=" * 60 + "\n")

    int8_images = []

    for i, img_id in enumerate(sample_ids):
        img_path = get_coco_image_path(images_dir, img_id)
        orig_img = cv2.imread(img_path)

        # PyTorch
        pt_dets, pt_time = run_pytorch(pt_model, img_path)
        pt_img = draw_detections(orig_img, pt_dets, colors)
        cv2.putText(pt_img, f"PyTorch FP32: {len(pt_dets)} objs, {pt_time:.1f}ms",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # TensorRT
        if engine:
            trt_dets, trt_time = run_trt(engine, img_path, conf_thres, iou_thres)
            trt_img = draw_detections(orig_img, trt_dets, colors)
            cv2.putText(trt_img, f"{engine_name}: {len(trt_dets)} objs, {trt_time:.1f}ms",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            trt_img = orig_img.copy()
            cv2.putText(trt_img, "TRT Engine Not Available", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Side-by-side
        combined = np.hstack((pt_img, trt_img))
        out_path = os.path.join(demo_dir, f"demo_sample_{i}.jpg")
        cv2.imwrite(out_path, combined)

        int8_images.append(cv2.resize(trt_img, (640, 640)))

        if engine:
            logger.info(f"Image {img_id}: PT={len(pt_dets)} objs ({pt_time:.1f}ms) | {engine_name}={len(trt_dets)} objs ({trt_time:.1f}ms)")
        else:
            logger.info(f"Image {img_id}: PT={len(pt_dets)} objs ({pt_time:.1f}ms) | TRT not available")

    if engine:
        engine.cleanup()

    # Create 2x4 grid
    if len(int8_images) == 8:
        row1 = np.hstack(int8_images[:4])
        row2 = np.hstack(int8_images[4:])
        grid = np.vstack((row1, row2))
        grid_path = os.path.join(demo_dir, "demo_grid_int8.jpg")
        cv2.imwrite(grid_path, grid)
        logger.info(f"Saved 2x4 grid to {grid_path}")


if __name__ == "__main__":
    main()
