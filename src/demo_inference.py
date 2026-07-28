"""
Script to run visual demo inference.
Compares PyTorch FP32 and TensorRT predictions side-by-side.

NOTE: PyTorch and TRT inferences are done in separate passes
to avoid CUDA context conflicts between torch and PyCUDA.
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


def main():
    parser = argparse.ArgumentParser(description="Run visual demo inference")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="Confidence threshold for visualization")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    args = parser.parse_args()

    config = load_config(args.config)
    demo_dir = config["paths"]["demo_dir"]
    # Clean old demo files to avoid stale outputs
    if os.path.exists(demo_dir):
        import glob
        for old_file in glob.glob(os.path.join(demo_dir, "demo_*")):
            os.remove(old_file)
    os.makedirs(demo_dir, exist_ok=True)

    ann_file = config["paths"]["coco_annotations"]
    images_dir = config["paths"]["coco_images"]

    if not os.path.exists(ann_file) or not os.path.exists(images_dir):
        logger.error("COCO dataset not found. Run evaluate_accuracy.py first.")
        sys.exit(1)

    image_ids = get_coco_image_ids(ann_file)
    random.seed(123)
    sample_ids = random.sample(image_ids, 8)

    colors = get_color_palette(len(COCO_CLASS_NAMES))
    conf_thres = args.conf_thres
    iou_thres = args.iou_thres

    # ================================================================
    # PASS 1: Run ALL PyTorch inferences first (uses torch CUDA context)
    # ================================================================
    logger.info("Pass 1: Running PyTorch inference on all samples...")
    pt_model = YOLO(config["model"]["weights"])
    pt_results = {}  # img_id -> (detections, time_ms)

    for img_id in sample_ids:
        img_path = get_coco_image_path(images_dir, img_id)
        t0 = time.perf_counter()
        results = pt_model(img_path, verbose=False)
        t1 = time.perf_counter()
        det = results[0].boxes.data.cpu().numpy()
        pt_results[img_id] = (det, (t1 - t0) * 1000)
        logger.info(f"  PyTorch img {img_id}: {len(det)} detections, {(t1-t0)*1000:.1f}ms")

    # Free PyTorch GPU memory before loading TRT
    del pt_model
    import torch
    torch.cuda.empty_cache()

    # ================================================================
    # PASS 2: Run ALL TRT inferences (uses PyCUDA context)
    # ================================================================
    engine = None
    engine_name = "TRT"
    trt_results = {}  # img_id -> (detections, time_ms)

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

    if engine:
        input_size = engine.get_input_shape()[-1]

        logger.info(f"Pass 2: Running {engine_name} inference on all samples...")

        for img_id in sample_ids:
            img_path = get_coco_image_path(images_dir, img_id)

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

            trt_results[img_id] = (dets, (t1 - t0) * 1000)
            logger.info(f"  {engine_name} img {img_id}: {len(dets)} detections, {(t1-t0)*1000:.1f}ms")

        engine.cleanup()
    else:
        logger.warning("No TRT engine available for demo")

    # ================================================================
    # PASS 3: Create side-by-side images
    # ================================================================
    logger.info("Pass 3: Creating side-by-side comparison images...")
    trt_grid_images = []

    for i, img_id in enumerate(sample_ids):
        img_path = get_coco_image_path(images_dir, img_id)
        orig_img = cv2.imread(img_path)

        # PyTorch side
        pt_dets, pt_time = pt_results[img_id]
        pt_img = draw_detections(orig_img, pt_dets, colors)
        cv2.putText(pt_img, f"PyTorch FP32: {len(pt_dets)} objs, {pt_time:.1f}ms",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # TRT side
        if img_id in trt_results:
            trt_dets, trt_time = trt_results[img_id]
            trt_img = draw_detections(orig_img, trt_dets, colors)
            cv2.putText(trt_img, f"{engine_name}: {len(trt_dets)} objs, {trt_time:.1f}ms",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            trt_img = orig_img.copy()
            cv2.putText(trt_img, "TRT Not Available", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Save side-by-side
        combined = np.hstack((pt_img, trt_img))
        out_path = os.path.join(demo_dir, f"demo_sample_{i}.jpg")
        cv2.imwrite(out_path, combined)

        trt_grid_images.append(cv2.resize(trt_img, (640, 640)))

        pt_n = len(pt_dets)
        trt_n = len(trt_dets) if img_id in trt_results else 0
        logger.info(f"  Sample {i}: PT={pt_n} objs | {engine_name}={trt_n} objs")

    # Create 2x4 grid
    if len(trt_grid_images) == 8:
        row1 = np.hstack(trt_grid_images[:4])
        row2 = np.hstack(trt_grid_images[4:])
        grid = np.vstack((row1, row2))
        grid_path = os.path.join(demo_dir, "demo_grid_trt.jpg")
        cv2.imwrite(grid_path, grid)
        logger.info(f"Saved 2x4 grid to {grid_path}")

    logger.info("Demo inference complete!")


if __name__ == "__main__":
    main()
