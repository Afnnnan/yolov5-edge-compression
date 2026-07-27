import os
import sys
import argparse
import onnx
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.common import load_config, setup_logger, Timer, ensure_dirs
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Export YOLOv5n-u to ONNX format")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--output", type=str, default=None, help="Override output path for the ONNX model")
    return parser.parse_args()

def main():
    args = parse_args()
    logger = setup_logger("export_onnx")
    config = load_config(args.config)
    
    ensure_dirs(config)

    model_name = config["model"]["name"]
    weights = config["model"]["weights"]
    input_size = config["model"]["input_size"]
    
    output_path = args.output if args.output else config["paths"]["onnx_model"]
    
    logger.info(f"Loading {weights}...")
    model = YOLO(weights)
    
    with Timer("ONNX Export", logger):
        logger.info(f"Exporting model to ONNX with opset=17, imgsz={input_size}...")
        exported_path = model.export(
            format="onnx",
            opset=17,
            simplify=True,
            imgsz=input_size,
            dynamic=False
        )
    
    # If the exported path doesn't match the desired output path, move it
    if exported_path != output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        os.rename(exported_path, output_path)
        logger.info(f"Moved exported model to {output_path}")
        exported_path = output_path
    
    logger.info("Validating ONNX model...")
    onnx_model = onnx.load(exported_path)
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model is valid.")
    
    # Print model info
    file_size_mb = os.path.getsize(exported_path) / (1024 * 1024)
    
    input_shapes = [
        {"name": inp.name, "shape": [d.dim_value for d in inp.type.tensor_type.shape.dim]}
        for inp in onnx_model.graph.input
    ]
    output_shapes = [
        {"name": out.name, "shape": [d.dim_value for d in out.type.tensor_type.shape.dim]}
        for out in onnx_model.graph.output
    ]
    
    # Parameter count (initializer elements)
    param_count = sum(
        [init.raw_data and len(init.raw_data) // 4 or sum(init.int64_data) or sum(init.float_data) for init in onnx_model.graph.initializer] # Approximation, accurate enough for general use
    )
    # A more robust way to count parameters in ONNX
    param_count = 0
    for tensor in onnx_model.graph.initializer:
        dims = tensor.dims
        size = 1
        for d in dims:
            size *= d
        param_count += size
    
    logger.info(f"ONNX Export Summary:")
    logger.info(f"  Path: {exported_path}")
    logger.info(f"  File Size: {file_size_mb:.2f} MB")
    logger.info(f"  Parameter Count: {param_count:,}")
    logger.info(f"  Inputs: {input_shapes}")
    logger.info(f"  Outputs: {output_shapes}")

if __name__ == '__main__':
    main()
