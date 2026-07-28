"""
Script to evaluate mAP on COCO val2017 for various YOLOv5n model variants.
Evaluates PyTorch FP32 and TensorRT FP32, FP16, and INT8 engines.

Uses ultralytics built-in val() for PyTorch (gold standard),
and custom postprocessing + pycocotools for TRT engines.
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

from utils.common import load_config, setup_logger, save_results, Timer, COCO_CLASS_IDS
from utils.coco_utils import (
    download_coco_val2017, preprocess_image, postprocess_yolov5,
    scale_boxes_to_original, get_coco_image_ids, get_coco_image_path
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


def evaluate_pytorch_builtin(model_path, coco_data_yaml, num_images=None):
    """
    Evaluate PyTorch model using ultralytics built-in val() method.
    This is the gold-standard evaluation and should match published numbers.
    """
    logger.info(f"Evaluating PyTorch model with ultralytics val(): {model_path}")
    model = YOLO(model_path)

    # Use ultralytics' built-in validation
    val_args = {
        "data": coco_data_yaml,
        "imgsz": 640,
        "batch": 16,
        "conf": 0.001,
        "iou": 0.7,  # NMS IoU threshold for COCO eval
        "verbose": False,
        "plots": False,
    }
    if num_images and num_images < 5000:
        val_args["max_det"] = 300

    results = model.val(**val_args)

    # ultralytics results object has .box attribute with mAP values
    map50 = results.box.map50  # mAP@0.5
    map50_95 = results.box.map  # mAP@0.5:0.95

    logger.info(f"PyTorch FP32: mAP@0.5={map50:.4f}, mAP@0.5:0.95={map50_95:.4f}")
    return {"mAP50": float(map50), "mAP50-95": float(map50_95)}


def evaluate_trt(engine_path, images_dir, ann_file, image_ids, num_images, conf_thres, iou_thres):
    """Evaluate TensorRT engine using custom postprocessing + pycocotools."""
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
        if not os.path.exists(img_path):
            continue

        input_tensor, _, meta = preprocess_image(img_path, input_size)

        outputs = engine.infer(input_tensor)
        out = outputs[0]

        # YOLOv5nu outputs (1, 84, 8400). Transpose to (1, 8400, 84) if needed.
        if out.shape[1] < out.shape[2]:
            out = np.transpose(out, (0, 2, 1))

        dets = postprocess_yolov5(out, conf_thres, iou_thres)[0]
        dets = scale_boxes_to_original(dets, meta)

        # Convert to COCO JSON format
        for det in dets:
            x1, y1, x2, y2, score, cls_id = det
            w = x2 - x1
            h = y2 - y1
            if w <= 0 or h <= 0:
                continue
            all_results.append({
                "image_id": int(img_id),
                "category_id": COCO_CLASS_IDS[int(cls_id)],
                "bbox": [round(float(x1), 2), round(float(y1), 2),
                         round(float(w), 2), round(float(h), 2)],
                "score": round(float(score), 5),
            })

    engine.cleanup()

    if not all_results:
        logger.warning(f"No detections from {engine_path}")
        return {"mAP50": 0.0, "mAP50-95": 0.0}

    # Compute mAP using pycocotools
    logger.info(f"Computing mAP with {len(all_results)} detections...")
    coco_gt = COCO(ann_file)
    coco_dt = coco_gt.loadRes(all_results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')

    # Only evaluate on the images we actually ran
    coco_eval.params.imgIds = list(image_ids[:num_images])
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return {
        "mAP50": float(coco_eval.stats[1]),
        "mAP50-95": float(coco_eval.stats[0]),
    }


def _create_coco_data_yaml(coco_dir, images_dir, ann_file):
    """Create a data yaml for ultralytics val()."""
    yaml_path = os.path.join(coco_dir, "coco_eval.yaml")

    # ultralytics expects paths relative to the yaml file or absolute
    abs_images = os.path.abspath(images_dir)
    yaml_content = f"""# COCO val2017 for evaluation
path: {os.path.abspath(coco_dir)}
val: {abs_images}
train: {abs_images}

names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: airplane
  5: bus
  6: train
  7: truck
  8: boat
  9: traffic light
  10: fire hydrant
  11: stop sign
  12: parking meter
  13: bench
  14: bird
  15: cat
  16: dog
  17: horse
  18: sheep
  19: cow
  20: elephant
  21: bear
  22: zebra
  23: giraffe
  24: backpack
  25: umbrella
  26: handbag
  27: tie
  28: suitcase
  29: frisbee
  30: skis
  31: snowboard
  32: sports ball
  33: kite
  34: baseball bat
  35: baseball glove
  36: skateboard
  37: surfboard
  38: tennis racket
  39: bottle
  40: wine glass
  41: cup
  42: fork
  43: knife
  44: spoon
  45: bowl
  46: banana
  47: apple
  48: sandwich
  49: orange
  50: broccoli
  51: carrot
  52: hot dog
  53: pizza
  54: donut
  55: cake
  56: chair
  57: couch
  58: potted plant
  59: bed
  60: dining table
  61: toilet
  62: tv
  63: laptop
  64: mouse
  65: remote
  66: keyboard
  67: cell phone
  68: microwave
  69: oven
  70: toaster
  71: sink
  72: refrigerator
  73: book
  74: clock
  75: vase
  76: scissors
  77: teddy bear
  78: hair drier
  79: toothbrush
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    return yaml_path


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

    results_summary = {}

    # PyTorch: use ultralytics built-in val() for gold-standard mAP
    if "pytorch" in args.models:
        coco_yaml = _create_coco_data_yaml(config["paths"]["coco_dir"], images_dir, ann_file)
        metrics = evaluate_pytorch_builtin(
            config["model"]["weights"], coco_yaml, num_images
        )
        results_summary["PyTorch FP32"] = metrics

    # TRT engines: custom evaluation with pycocotools
    trt_variants = {
        "trt_fp32": "fp32_engine",
        "trt_fp16": "fp16_engine",
        "trt_int8": "int8_engine"
    }

    for model_name, path_key in trt_variants.items():
        if model_name in args.models:
            engine_path = config["paths"].get(path_key)
            if engine_path:
                metrics = evaluate_trt(
                    engine_path, images_dir, ann_file, image_ids,
                    num_images, conf_thres, iou_thres
                )
                if metrics is not None:
                    results_summary[model_name.upper().replace("_", " ")] = metrics

    print("\n" + "=" * 50)
    print(f"{'Model':<20} | {'mAP@0.5':<10} | {'mAP@0.5:0.95'}")
    print("=" * 50)
    for model_name, metrics in results_summary.items():
        print(f"{model_name:<20} | {metrics['mAP50']:<10.4f} | {metrics['mAP50-95']:.4f}")
    print("=" * 50)

    out_path = os.path.join(config["paths"]["results_dir"], "accuracy_results.json")
    save_results(results_summary, out_path)


if __name__ == "__main__":
    main()
