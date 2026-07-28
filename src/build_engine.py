"""
Build TensorRT engines from an ONNX model.

Supports three precision modes:
  - FP32: Full precision baseline
  - FP16: Half precision (free speedup on T4 Tensor Cores)
  - INT8: Post-training quantization with calibration

Uses a hybrid approach for maximum compatibility:
  - FP32/FP16: TensorRT Python API (trt.Builder + OnnxParser)
  - INT8: Ultralytics export (handles calibration internally, works across TRT versions)
  - Fallback: trtexec command-line tool
"""

import os
import sys
import shutil
import argparse
import subprocess
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.common import load_config, setup_logger, Timer, ensure_dirs

logger = setup_logger("build_engine")

# Check TensorRT availability
try:
    import tensorrt as trt
    HAS_TRT = True
    TRT_VERSION = trt.__version__
    logger.info(f"TensorRT {TRT_VERSION} available")
except ImportError:
    HAS_TRT = False
    TRT_VERSION = None
    logger.warning("TensorRT not available")


def parse_args():
    parser = argparse.ArgumentParser(description="Build TensorRT engine from ONNX model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--precision", type=str, default="all",
                        choices=["fp32", "fp16", "int8", "all"],
                        help="Precision to build (default: all)")
    return parser.parse_args()


# =============================================================================
# Strategy 1: TensorRT Python API (for FP32 and FP16)
# =============================================================================

def build_engine_trt_api(onnx_path, output_path, precision="fp32"):
    """Build engine using TensorRT Python API. Works for FP32 and FP16."""
    if not HAS_TRT:
        raise RuntimeError("TensorRT not available")

    logger.info(f"Building {precision.upper()} engine using TRT Python API")

    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)

    # Create network with explicit batch
    network_flags = 0
    if hasattr(trt, 'NetworkDefinitionCreationFlag'):
        if hasattr(trt.NetworkDefinitionCreationFlag, 'EXPLICIT_BATCH'):
            network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config = builder.create_builder_config()

    # 2GB workspace
    if hasattr(config, 'set_memory_pool_limit'):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    else:
        config.max_workspace_size = 2 << 30  # Legacy API

    # Parse ONNX
    with open(onnx_path, "rb") as model:
        if not parser.parse(model.read()):
            for i in range(parser.num_errors):
                logger.error(f"ONNX Parse Error: {parser.get_error(i)}")
            raise RuntimeError("Failed to parse ONNX file")

    logger.info(f"ONNX parsed successfully: {network.num_layers} layers")

    # Set precision flags
    if precision == "fp16":
        if hasattr(trt, 'BuilderFlag') and hasattr(trt.BuilderFlag, 'FP16'):
            config.set_flag(trt.BuilderFlag.FP16)
        logger.info("FP16 mode enabled")

    # Build engine
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError(f"Failed to build {precision.upper()} engine")

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(serialized_engine)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"✅ {precision.upper()} engine saved: {output_path} ({size_mb:.1f} MB)")
    return True


# =============================================================================
# Strategy 2: Ultralytics Export (for INT8 — handles calibration internally)
# =============================================================================

def build_engine_ultralytics(weights, output_path, precision="int8", config=None):
    """
    Build engine using ultralytics export.
    Best for INT8 as it handles calibration automatically.
    Also works for FP32/FP16 as a fallback.
    """
    logger.info(f"Building {precision.upper()} engine using ultralytics export")

    from ultralytics import YOLO
    model = YOLO(weights)

    input_size = config["model"]["input_size"] if config else 640

    export_kwargs = {
        "format": "engine",
        "imgsz": input_size,
        "device": 0,
        "simplify": True,
    }

    if precision == "fp16":
        export_kwargs["half"] = True
    elif precision == "int8":
        export_kwargs["int8"] = True
        # Create a COCO data yaml for calibration
        coco_yaml = _create_coco_yaml(config)
        if coco_yaml:
            export_kwargs["data"] = coco_yaml

    exported_path = model.export(**export_kwargs)
    logger.info(f"Ultralytics exported engine to: {exported_path}")

    # Move to desired output path
    if exported_path and os.path.exists(exported_path) and str(exported_path) != str(output_path):
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        shutil.move(str(exported_path), str(output_path))
        logger.info(f"Moved engine to: {output_path}")

    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(f"✅ {precision.upper()} engine saved: {output_path} ({size_mb:.1f} MB)")
        return True

    return False


def _create_coco_yaml(config):
    """Create a COCO data yaml for ultralytics INT8 calibration."""
    if not config:
        return "coco.yaml"  # ultralytics built-in

    coco_dir = config["paths"].get("coco_dir", "data/coco")
    images_dir = config["paths"].get("coco_images", "data/coco/val2017")

    if not os.path.exists(images_dir):
        logger.warning(f"COCO images not found at {images_dir}, using ultralytics built-in coco.yaml")
        return "coco.yaml"

    # Create a custom data yaml pointing to our downloaded COCO
    yaml_path = os.path.join(coco_dir, "coco_calib.yaml")
    abs_images = os.path.abspath(images_dir)

    yaml_content = f"""# COCO val2017 for INT8 calibration
path: {os.path.abspath(coco_dir)}
val: {abs_images}
train: {abs_images}

# COCO 80 classes
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
    os.makedirs(os.path.dirname(yaml_path) or ".", exist_ok=True)
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    logger.info(f"Created calibration data yaml: {yaml_path}")
    return yaml_path


# =============================================================================
# Strategy 3: trtexec CLI (fallback for FP32/FP16)
# =============================================================================

def build_engine_trtexec(onnx_path, output_path, precision="fp32"):
    """Build engine using trtexec command-line tool."""
    trtexec = shutil.which("trtexec")
    if not trtexec:
        # Check common paths
        for path in ["/usr/src/tensorrt/bin/trtexec", "/usr/local/bin/trtexec"]:
            if os.path.exists(path):
                trtexec = path
                break

    if not trtexec:
        logger.warning("trtexec not found in PATH")
        return False

    logger.info(f"Building {precision.upper()} engine using trtexec")

    cmd = [trtexec, f"--onnx={onnx_path}", f"--saveEngine={output_path}"]
    if precision == "fp16":
        cmd.append("--fp16")
    elif precision == "int8":
        cmd.extend(["--int8", "--fp16"])  # FP16 fallback for INT8

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode == 0 and os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(f"✅ {precision.upper()} engine saved: {output_path} ({size_mb:.1f} MB)")
        return True

    logger.error(f"trtexec failed: {result.stderr[-500:] if result.stderr else 'unknown error'}")
    return False


# =============================================================================
# Main: Try strategies in order of preference
# =============================================================================

def build_engine(config, precision="fp32"):
    """Build a TensorRT engine with automatic strategy selection."""
    onnx_path = config["paths"]["onnx_model"]
    output_path = config["paths"].get(f"{precision}_engine", f"outputs/yolov5n_{precision}.engine")
    weights = config["model"]["weights"]

    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}. Run export_onnx.py first.")

    logger.info(f"\n{'='*60}")
    logger.info(f"Building {precision.upper()} TensorRT Engine")
    logger.info(f"{'='*60}")

    if precision == "int8":
        # INT8 always uses ultralytics (handles calibration automatically)
        with Timer(f"Build {precision.upper()} Engine", logger):
            success = build_engine_ultralytics(weights, output_path, precision, config)
        if not success:
            logger.error(f"Failed to build {precision.upper()} engine with all strategies")
        return success

    # FP32 / FP16: try TRT Python API first, then trtexec, then ultralytics
    strategies = [
        ("TRT Python API", lambda: build_engine_trt_api(onnx_path, output_path, precision)),
        ("trtexec CLI", lambda: build_engine_trtexec(onnx_path, output_path, precision)),
        ("Ultralytics export", lambda: build_engine_ultralytics(weights, output_path, precision, config)),
    ]

    for name, strategy_fn in strategies:
        try:
            with Timer(f"Build {precision.upper()} Engine ({name})", logger):
                success = strategy_fn()
            if success:
                return True
            logger.warning(f"{name} did not produce engine, trying next strategy...")
        except Exception as e:
            logger.warning(f"{name} failed: {e}. Trying next strategy...")

    logger.error(f"Failed to build {precision.upper()} engine with all strategies")
    return False


def main():
    args = parse_args()
    config = load_config(args.config)
    ensure_dirs(config)

    precisions = ["fp32", "fp16", "int8"] if args.precision == "all" else [args.precision]

    results = {}
    for prec in precisions:
        results[prec] = build_engine(config, prec)

    # Summary
    print(f"\n{'='*60}")
    print("ENGINE BUILD SUMMARY")
    print(f"{'='*60}")
    for prec, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        path = f"outputs/yolov5n_{prec}.engine"
        size = ""
        if os.path.exists(path):
            size = f" ({os.path.getsize(path) / (1024**2):.1f} MB)"
        print(f"  {prec.upper()}: {status}{size}")


if __name__ == '__main__':
    main()
