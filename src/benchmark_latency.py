import os
import sys
import argparse
import numpy as np
import torch
from tabulate import tabulate

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.common import load_config, setup_logger, save_results, ensure_dirs
from utils.trt_inference import TRTInference, CUDATimer, HAS_TENSORRT
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark model latency and throughput")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    return parser.parse_args()

def calculate_metrics(latencies, batch_size=1):
    latencies = np.array(latencies)
    mean_lat = np.mean(latencies)
    metrics = {
        "mean_ms": mean_lat,
        "median_ms": np.median(latencies),
        "p95_ms": np.percentile(latencies, 95),
        "p99_ms": np.percentile(latencies, 99),
        "std_ms": np.std(latencies),
        "fps": (batch_size * 1000.0) / mean_lat if mean_lat > 0 else 0
    }
    return metrics

def benchmark_pytorch(logger, config, device="cuda"):
    model_name = config["model"]["name"]
    weights = config["model"]["weights"]
    input_size = config["model"]["input_size"]
    batch_size = config["model"]["batch_size"]
    warmup = config["benchmark"]["warmup_iterations"]
    measure = config["benchmark"]["measure_iterations"]
    
    logger.info(f"Benchmarking PyTorch model: {weights} on {device}")
    
    model = YOLO(weights)
    model.model.to(device)
    model.model.eval()
    
    # Input tensor
    input_tensor = torch.randn(batch_size, 3, input_size, input_size).to(device)
    
    latencies = []
    
    with torch.no_grad():
        logger.info(f"Warming up for {warmup} iterations...")
        for _ in range(warmup):
            _ = model.model(input_tensor)
            
        if device == "cuda":
            torch.cuda.synchronize()
            
        logger.info(f"Measuring for {measure} iterations...")
        for _ in range(measure):
            if device == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                
                _ = model.model(input_tensor)
                
                end_event.record()
                torch.cuda.synchronize()
                latencies.append(start_event.elapsed_time(end_event))
            else:
                import time
                start = time.perf_counter()
                _ = model.model(input_tensor)
                latencies.append((time.perf_counter() - start) * 1000.0)

    return calculate_metrics(latencies, batch_size)

def benchmark_trt(logger, engine_path, config):
    input_size = config["model"]["input_size"]
    batch_size = config["model"]["batch_size"]
    warmup = config["benchmark"]["warmup_iterations"]
    measure = config["benchmark"]["measure_iterations"]
    
    logger.info(f"Benchmarking TensorRT engine: {engine_path}")
    
    if not os.path.exists(engine_path):
        logger.warning(f"Engine not found: {engine_path}. Skipping.")
        return None

    try:
        engine = TRTInference(engine_path)
    except Exception as e:
        logger.warning(f"Failed to load engine {engine_path}: {e}. Skipping.")
        return None
    timer = CUDATimer()
    
    # Generate random input data
    input_data = np.random.randn(batch_size, 3, input_size, input_size).astype(np.float32)
    
    latencies = []
    
    logger.info(f"Warming up for {warmup} iterations...")
    for _ in range(warmup):
        _ = engine.infer(input_data)
        
    logger.info(f"Measuring for {measure} iterations...")
    for _ in range(measure):
        timer.start(engine.stream)
        _ = engine.infer(input_data)
        timer.stop(engine.stream)
        latencies.append(timer.elapsed_ms())
        
    engine.cleanup()
    return calculate_metrics(latencies, batch_size)

def main():
    args = parse_args()
    logger = setup_logger("benchmark_latency")
    config = load_config(args.config)
    ensure_dirs(config)
    
    results = {}
    
    # 1. PyTorch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pt_metrics = benchmark_pytorch(logger, config, device)
    results["PyTorch FP32"] = pt_metrics

    # 2. TensorRT
    if HAS_TENSORRT:
        precisions = ["fp32", "fp16", "int8"]
        for prec in precisions:
            engine_path = config["paths"].get(f"{prec}_engine", f"outputs/yolov5n_{prec}.engine")
            trt_metrics = benchmark_trt(logger, engine_path, config)
            if trt_metrics:
                results[f"TRT {prec.upper()}"] = trt_metrics
    else:
        logger.warning("TensorRT not available, skipping TRT benchmarks.")
        
    # Save results
    out_file = os.path.join(config["paths"]["results_dir"], "benchmark_latency.json")
    save_results(results, out_file)
    
    # Print table
    headers = ["Model", "Mean (ms)", "Median (ms)", "P95 (ms)", "P99 (ms)", "Std (ms)", "FPS"]
    table_data = []
    for name, metrics in results.items():
        row = [
            name,
            f"{metrics['mean_ms']:.2f}",
            f"{metrics['median_ms']:.2f}",
            f"{metrics['p95_ms']:.2f}",
            f"{metrics['p99_ms']:.2f}",
            f"{metrics['std_ms']:.2f}",
            f"{metrics['fps']:.2f}"
        ]
        table_data.append(row)
        
    print("\nLatency Benchmark Results:")
    print(tabulate(table_data, headers=headers, tablefmt="pretty"))

if __name__ == '__main__':
    main()
