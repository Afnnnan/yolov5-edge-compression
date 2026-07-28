"""
Script to evaluate mAP on COCO val2017 for various YOLOv5n model variants.
Evaluates PyTorch FP32 and TensorRT FP32, FP16, and INT8 engines.
"""

import os
import sys
import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.common import load_config, setup_logger, save_results, Timer
from utils.coco_utils import (
    download_coco_val2017, preprocess_image, postprocess_yolov5,
    scale_boxes_to_original, detections_to_coco_json, get_coco_image_ids, get_coco_image_path
)

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError:
    print("Please install pycocotools: pip install pycocotools")
    sys.exit(1)

import torch
from ultralytics import YOLO

logger = setup_logger("evaluate_accuracy")


def evaluate_pytorch(model_path, images_dir, image_ids, num_images, conf_thres, iou_thres):
    """Evaluate PyTorch model using ultralytics YOLO."""
    logger.info(f"Loading PyTorch model: {model_path}")
    model = YOLO(model_path)
    all_results = []
    
    for img_id in tqdm(image_ids[:num_images], desc="PyTorch FP32"):
        img_path = get_coco_image_path(images_dir, img_id)
        # Use low conf threshold so pycocotools can compute full precision-recall curve
        results = model(img_path, verbose=False, conf=conf_thres, iou=iou_thres, max_det=300)
        # boxes.data is [x1, y1, x2, y2, conf, cls]
        det = results[0].boxes.data.cpu().numpy()
        
        coco_res = detections_to_coco_json(det, img_id)
        all_results.extend(coco_res)
        
    return all_results


def evaluate_trt(engine_path, images_dir, image_ids, num_images, conf_thres, iou_thres):
    """Evaluate TensorRT engine."""
    if not os.path.exists(engine_path):
        logger.warning(f"Engine not found (skipping): {engine_path}")
        return None
        
    try:
        from utils.trt_inference import TRTInference
        engine = TRTInference(engine_path)
    except Exception as e:
        logger.error(f"Failed to load engine {engine_path}: {e}")
        return None
        
    input_size = engine.get_input_shape()[-1]
    all_results = []
    
    for img_id in tqdm(image_ids[:num_images], desc=f"TRT {os.path.basename(engine_path)}"):
        img_path = get_coco_image_path(images_dir, img_id)
        input_tensor, _, meta = preprocess_image(img_path, input_size)
        
        outputs = engine.infer(input_tensor)
        out = outputs[0]
        
        # YOLOv5nu typically outputs (1, 84, 8400). Transpose to (1, 8400, 84) if needed.
        if out.shape[1] < out.shape[2]:
            out = np.transpose(out, (0, 2, 1))
            
        dets = postprocess_yolov5(out, conf_thres, iou_thres)[0]
        dets = scale_boxes_to_original(dets, meta)
        
        coco_res = detections_to_coco_json(dets, img_id)
        all_results.extend(coco_res)
        
    engine.cleanup()
    return all_results


def compute_map(coco_gt, coco_res_list):
    """Compute mAP using pycocotools."""
    if not coco_res_list:
        return {"mAP50": 0.0, "mAP50-95": 0.0}
        
    coco_dt = coco_gt.loadRes(coco_res_list)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    return {
        "mAP50": coco_eval.stats[1],
        "mAP50-95": coco_eval.stats[0]
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv5n accuracy on COCO.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--num-images", type=int, default=None,
                        help="Number of images to evaluate")
    parser.add_argument("--models", nargs="+", default=["pytorch", "trt_fp32", "trt_fp16", "trt_int8"],
                        help="Models to evaluate")
    args = parser.parse_args()

    config = load_config(args.config)
    num_images = args.num_images or config["evaluation"]["num_images"]
    conf_thres = config["evaluation"]["confidence_threshold"]
    iou_thres = config["evaluation"]["nms_iou_threshold"]
    
    images_dir, ann_file = download_coco_val2017(config["paths"]["coco_dir"])
    image_ids = get_coco_image_ids(ann_file)
    
    logger.info("Loading COCO annotations...")
    coco_gt = COCO(ann_file)
    
    results_summary = {}
    
    if "pytorch" in args.models:
        model_path = config["model"]["weights"]
        preds = evaluate_pytorch(model_path, images_dir, image_ids, num_images, conf_thres, iou_thres)
        metrics = compute_map(coco_gt, preds)
        results_summary["PyTorch FP32"] = metrics
        
    trt_variants = {
        "trt_fp32": "fp32_engine",
        "trt_fp16": "fp16_engine",
        "trt_int8": "int8_engine"
    }
    
    for model_name, path_key in trt_variants.items():
        if model_name in args.models:
            engine_path = config["paths"].get(path_key)
            if engine_path:
                preds = evaluate_trt(engine_path, images_dir, image_ids, num_images, conf_thres, iou_thres)
                if preds is not None:
                    metrics = compute_map(coco_gt, preds)
                    results_summary[model_name.upper().replace("_", " ")] = metrics

    print("\n" + "="*50)
    print(f"{'Model':<20} | {'mAP@0.5':<10} | {'mAP@0.5:0.95'}")
    print("="*50)
    for model_name, metrics in results_summary.items():
        print(f"{model_name:<20} | {metrics['mAP50']:<10.4f} | {metrics['mAP50-95']:.4f}")
    print("="*50)

    out_path = os.path.join(config["paths"]["results_dir"], "accuracy_results.json")
    save_results(results_summary, out_path)


if __name__ == "__main__":
    main()
