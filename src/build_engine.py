import os
import sys
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.common import load_config, setup_logger, Timer, ensure_dirs
from utils.coco_utils import CalibrationDataLoader
from utils.calibrator import YOLOv5EntropyCalibrator, HAS_TENSORRT

if HAS_TENSORRT:
    import tensorrt as trt
else:
    print("WARNING: TensorRT is not available. Script will fail if executed.")

def parse_args():
    parser = argparse.ArgumentParser(description="Build TensorRT engine from ONNX model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--precision", type=str, default="all", choices=["fp32", "fp16", "int8", "all"], help="Precision to build")
    return parser.parse_args()

def build_engine(logger, config, precision="fp32"):
    onnx_path = config["paths"]["onnx_model"]
    output_path = config["paths"].get(f"{precision}_engine", f"outputs/yolov5n_{precision}.engine")
    
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        
    logger.info(f"Building {precision.upper()} engine from {onnx_path}")
    
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    builder_config = builder.create_builder_config()
    
    # 2GB workspace
    workspace_size = 2 * (1 << 30)
    builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)
    
    # Parse ONNX
    with open(onnx_path, "rb") as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                logger.error(parser.get_error(error))
            raise RuntimeError("Failed to parse ONNX file")
            
    # Set precision flags
    if precision == "fp16":
        builder_config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        builder_config.set_flag(trt.BuilderFlag.FP16) # fallback
        builder_config.set_flag(trt.BuilderFlag.INT8)
        
        # Setup calibrator
        input_size = config["model"]["input_size"]
        batch_size = config["model"]["batch_size"]
        calib_images = config["calibration"]["num_images"]
        cache_file = config["calibration"]["cache_file"]
        coco_images = config["paths"]["coco_images"]
        
        logger.info(f"Setting up INT8 Calibrator with {calib_images} images")
        data_loader = CalibrationDataLoader(
            images_dir=coco_images,
            num_images=calib_images,
            input_size=input_size,
            batch_size=batch_size
        )
        
        calibrator = YOLOv5EntropyCalibrator(
            data_loader=data_loader,
            cache_file=cache_file,
            input_shape=(batch_size, 3, input_size, input_size)
        )
        builder_config.int8_calibrator = calibrator
    
    with Timer(f"Build {precision.upper()} Engine", logger):
        engine = builder.build_serialized_network(network, builder_config)
        if engine is None:
            raise RuntimeError(f"Failed to build {precision.upper()} engine")
            
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(engine)
            
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"Successfully built {precision.upper()} engine.")
    logger.info(f"Engine saved to: {output_path} ({file_size_mb:.2f} MB)")

def main():
    args = parse_args()
    logger = setup_logger("build_engine")
    config = load_config(args.config)
    ensure_dirs(config)
    
    if not HAS_TENSORRT:
        logger.error("TensorRT is required to build engines.")
        return
        
    precisions_to_build = ["fp32", "fp16", "int8"] if args.precision == "all" else [args.precision]
    
    for prec in precisions_to_build:
        build_engine(logger, config, prec)

if __name__ == '__main__':
    main()
