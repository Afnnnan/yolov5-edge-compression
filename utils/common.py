"""
Common utilities for the model compression pipeline.

Provides:
- YAML config loading
- Logging setup
- Timing utilities
- GPU info helpers
- Path management
"""

import os
import sys
import time
import logging
import functools
from pathlib import Path

import yaml
import numpy as np


# =============================================================================
# Configuration
# =============================================================================

def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load pipeline configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def ensure_dirs(config: dict):
    """Create all output directories specified in config."""
    paths = config.get("paths", {})
    for key in ["output_dir", "results_dir", "demo_dir"]:
        if key in paths:
            os.makedirs(paths[key], exist_ok=True)


# =============================================================================
# Logging
# =============================================================================

def setup_logger(name: str = "pipeline", level: int = logging.INFO) -> logging.Logger:
    """Create a formatted logger."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # avoid duplicate handlers

    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s | %(name)-12s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


# =============================================================================
# Timing
# =============================================================================

class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, name: str = "", logger: logging.Logger = None):
        self.name = name
        self.logger = logger
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start
        msg = f"[{self.name}] completed in {self.elapsed:.2f}s"
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)


def timed(func):
    """Decorator to log function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"[{func.__name__}] completed in {elapsed:.2f}s")
        return result
    return wrapper


# =============================================================================
# GPU Utilities
# =============================================================================

def print_gpu_info():
    """Print GPU information using nvidia-smi and torch."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
            print(f"GPU: {gpu_name}")
            print(f"GPU Memory: {gpu_mem:.1f} GB")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"PyTorch Version: {torch.__version__}")
        else:
            print("WARNING: No CUDA GPU available!")
    except ImportError:
        print("WARNING: PyTorch not installed, cannot query GPU info")

    # Also try nvidia-smi
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            print(f"nvidia-smi: {result.stdout.strip()}")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


def get_tensorrt_version() -> str:
    """Return TensorRT version string."""
    try:
        import tensorrt as trt
        return trt.__version__
    except ImportError:
        return "not installed"


# =============================================================================
# Results I/O
# =============================================================================

def save_results(results: dict, filepath: str):
    """Save benchmark/evaluation results to JSON."""
    import json
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=_json_serializer)
    print(f"Results saved to: {filepath}")


def load_results(filepath: str) -> dict:
    """Load results from JSON file."""
    import json
    with open(filepath, "r") as f:
        return json.load(f)


def _json_serializer(obj):
    """Handle numpy types in JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# =============================================================================
# COCO Class Mapping
# =============================================================================

# YOLOv5 uses 0-indexed class IDs (0-79), COCO uses specific category IDs
# This is the standard mapping from YOLOv5 class index → COCO category ID
COCO_CLASS_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
]

COCO_CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


def yolo_class_to_coco_id(yolo_class_idx: int) -> int:
    """Convert YOLOv5 0-indexed class to COCO category ID."""
    return COCO_CLASS_IDS[yolo_class_idx]
