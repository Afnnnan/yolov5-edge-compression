"""
=============================================================================
YOLOv5n Model Compression Pipeline - Kaggle Notebook
=============================================================================

Complete end-to-end pipeline for compressing YOLOv5n via:
  PyTorch -> ONNX -> TensorRT (FP32 / FP16 / INT8)

Run this on Kaggle with a T4 GPU accelerator.
Each section is separated by `# %%` for easy conversion to notebook cells.

Instructions:
  1. Create a new Kaggle notebook
  2. Enable GPU accelerator (T4)
  3. Enable internet access
  4. Paste each cell section into the notebook
  5. Run cells sequentially

Author: Model Compression Pipeline
Target: Kaggle T4 GPU (NVIDIA Tesla T4, 16GB GDDR6)
=============================================================================
"""

# %% [markdown]
# # YOLOv5n Model Compression Pipeline
#
# **Objective**: Compress YOLOv5-nano for edge deployment using:
# - ONNX export
# - TensorRT optimization (FP32, FP16, INT8)
# - Post-Training Quantization (PTQ)
#
# **Hardware**: Kaggle T4 GPU (NVIDIA Tesla T4)
#
# **Pipeline**:
# 1. Environment Setup
# 2. Data Download (COCO val2017)
# 3. ONNX Export
# 4. TensorRT Engine Build
# 5. Latency & Throughput Benchmarking
# 6. Accuracy Evaluation (mAP)
# 7. Visualization
# 8. Demo Inference

# %% [markdown]
# ## 1. Environment Setup

# %%
# === 1.1 Install Dependencies ===
import subprocess
import sys

def run(cmd):
    """Run a shell command and print output."""
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout[-2000:])  # Print last 2000 chars
    if result.returncode != 0 and result.stderr:
        print(f"STDERR: {result.stderr[-1000:]}")
    return result.returncode

# Install TensorRT from NVIDIA PyPI
run("pip install -q --extra-index-url https://pypi.nvidia.com tensorrt-cu12")

# Install PyCUDA for GPU memory management
run("pip install -q pycuda")

# Install ultralytics for YOLOv5
run("pip install -q ultralytics>=8.3.0")

# Install other dependencies
run("pip install -q pycocotools>=2.0.6 seaborn tabulate onnx")

# %%
# === 1.2 Clone Project Repository ===
import os

REPO_URL = "https://github.com/Afnnnan/yolov5-edge-compression.git"  # GitHub repo
PROJECT_DIR = "/kaggle/working/topcoder"

if not os.path.exists(PROJECT_DIR):
    run(f"git clone {REPO_URL} {PROJECT_DIR}")
else:
    print(f"Project already cloned at {PROJECT_DIR}")
    run(f"cd {PROJECT_DIR} && git pull")

os.chdir(PROJECT_DIR)
print(f"Working directory: {os.getcwd()}")

# %%
# === 1.3 Verify GPU & Environment ===
import torch
import numpy as np

print("=" * 60)
print("ENVIRONMENT VERIFICATION")
print("=" * 60)

# GPU Check
assert torch.cuda.is_available(), "ERROR: No GPU available! Enable GPU in Kaggle settings."
gpu_name = torch.cuda.get_device_name(0)
gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
print(f"[OK] GPU: {gpu_name} ({gpu_mem:.1f} GB)")
print(f"[OK] CUDA: {torch.version.cuda}")
print(f"[OK] PyTorch: {torch.__version__}")

# TensorRT Check
try:
    import tensorrt as trt
    print(f"[OK] TensorRT: {trt.__version__}")
except ImportError:
    print("[ERROR] TensorRT not found! Install failed.")

# PyCUDA Check
try:
    import pycuda.driver as cuda_drv
    import pycuda.autoinit
    print(f"[OK] PyCUDA available")
except ImportError:
    print("[ERROR] PyCUDA not found!")

# ONNX Check
try:
    import onnx
    print(f"[OK] ONNX: {onnx.__version__}")
except ImportError:
    print("[ERROR] ONNX not found!")

print("=" * 60)

# %% [markdown]
# ## 2. Download COCO val2017 Dataset

# %%
# === 2.1 Download COCO val2017 ===
# This downloads ~1GB of images + annotations

COCO_DIR = "data/coco"
os.makedirs(COCO_DIR, exist_ok=True)

images_dir = os.path.join(COCO_DIR, "val2017")
ann_dir = os.path.join(COCO_DIR, "annotations")
ann_file = os.path.join(ann_dir, "instances_val2017.json")

if not os.path.exists(images_dir) or len(os.listdir(images_dir)) < 5000:
    print("Downloading COCO val2017 images (~1 GB)...")
    run(f"wget -q http://images.cocodataset.org/zips/val2017.zip -O {COCO_DIR}/val2017.zip")
    run(f"unzip -q -o {COCO_DIR}/val2017.zip -d {COCO_DIR}")
    run(f"rm {COCO_DIR}/val2017.zip")
    print(f"[OK] Images: {len(os.listdir(images_dir))} files")
else:
    print(f"[OK] Images already downloaded: {len(os.listdir(images_dir))} files")

if not os.path.exists(ann_file):
    print("Downloading COCO annotations...")
    run(f"wget -q http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O {COCO_DIR}/ann.zip")
    run(f"unzip -q -o {COCO_DIR}/ann.zip -d {COCO_DIR}")
    run(f"rm {COCO_DIR}/ann.zip")
    print("[OK] Annotations downloaded")
else:
    print("[OK] Annotations already downloaded")

# %% [markdown]
# ## 3. Step 1 - Export YOLOv5n to ONNX

# %%
# === 3.1 ONNX Export ===
print("=" * 60)
print("STEP 1: ONNX EXPORT")
print("=" * 60)

run("python src/export_onnx.py --config configs/config.yaml")

# Verify the export
onnx_path = "outputs/yolov5n.onnx"
if os.path.exists(onnx_path):
    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"\n[OK] ONNX model exported: {onnx_path} ({size_mb:.1f} MB)")
else:
    print("[ERROR] ONNX export failed!")

# %% [markdown]
# ## 4. Step 2 - Build TensorRT Engines

# %%
# === 4.1 Build All TRT Engines (FP32, FP16, INT8) ===
print("=" * 60)
print("STEP 2: BUILD TENSORRT ENGINES")
print("=" * 60)

# Build FP32 engine
print("\n--- Building FP32 Engine ---")
run("python src/build_engine.py --config configs/config.yaml --precision fp32")

# Build FP16 engine
print("\n--- Building FP16 Engine ---")
run("python src/build_engine.py --config configs/config.yaml --precision fp16")

# Build INT8 engine (this runs calibration - takes a few minutes)
print("\n--- Building INT8 Engine (with calibration) ---")
run("python src/build_engine.py --config configs/config.yaml --precision int8")

# Verify engines
print("\n--- Engine Files ---")
for precision in ["fp32", "fp16", "int8"]:
    path = f"outputs/yolov5n_{precision}.engine"
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  [OK] {precision.upper()}: {path} ({size_mb:.1f} MB)")
    else:
        print(f"  [ERROR] {precision.upper()}: not found")

# %% [markdown]
# ## 5. Step 3 - Latency & Throughput Benchmarking

# %%
# === 5.1 Run Benchmarks ===
print("=" * 60)
print("STEP 3: LATENCY & THROUGHPUT BENCHMARKING")
print("=" * 60)

run("python src/benchmark_latency.py --config configs/config.yaml")

# %%
# === 5.2 Display Results ===
import json

results_path = "outputs/results/benchmark_latency.json"
if os.path.exists(results_path):
    with open(results_path) as f:
        bench_results = json.load(f)

    print("\nBenchmark Results Summary:")
    print("-" * 80)
    print(f"{'Model':<20} {'Mean (ms)':<12} {'Median (ms)':<12} {'P95 (ms)':<12} {'FPS':<10}")
    print("-" * 80)
    for model_name, metrics in bench_results.items():
        if isinstance(metrics, dict):
            print(
                f"{model_name:<20} "
                f"{metrics.get('mean_ms', 0):<12.2f} "
                f"{metrics.get('median_ms', 0):<12.2f} "
                f"{metrics.get('p95_ms', 0):<12.2f} "
                f"{metrics.get('fps', 0):<10.1f}"
            )
    print("-" * 80)

# %% [markdown]
# ## 6. Step 4 - Accuracy Evaluation (mAP)

# %%
# === 6.1 Run mAP Evaluation ===
# NOTE: Full evaluation on 5000 images takes 30-60 minutes.
# Use --num-images 1000 for a faster run (~10-15 min).
print("=" * 60)
print("STEP 4: ACCURACY EVALUATION (mAP)")
print("=" * 60)

# Using 1000 images for faster evaluation. Change to 5000 for full eval.
NUM_EVAL_IMAGES = 1000
run(f"python src/evaluate_accuracy.py --config configs/config.yaml --num-images {NUM_EVAL_IMAGES}")

# %%
# === 6.2 Display Accuracy Results ===
acc_path = "outputs/results/accuracy_results.json"
if os.path.exists(acc_path):
    with open(acc_path) as f:
        acc_results = json.load(f)

    print("\nAccuracy Results Summary:")
    print("-" * 60)
    print(f"{'Model':<20} {'mAP@0.5':<12} {'mAP@0.5:0.95':<15}")
    print("-" * 60)
    for model_name, metrics in acc_results.items():
        if isinstance(metrics, dict):
            print(
                f"{model_name:<20} "
                f"{metrics.get('mAP50', 0):<12.4f} "
                f"{metrics.get('mAP50-95', 0):<15.4f}"
            )
    print("-" * 60)

# %% [markdown]
# ## 7. Step 5 - Visualization

# %%
# === 7.1 Generate Comparison Charts ===
print("=" * 60)
print("STEP 5: GENERATE COMPARISON CHARTS")
print("=" * 60)

run("python src/visualize_results.py --config configs/config.yaml")

# %%
# === 7.2 Display Charts ===
from IPython.display import Image, display
import glob

chart_files = sorted(glob.glob("outputs/results/*.png"))
for chart_file in chart_files:
    print(f"\n[Chart] {os.path.basename(chart_file)}")
    display(Image(filename=chart_file, width=800))

# %% [markdown]
# ## 8. Step 6 - Demo Inference

# %%
# === 8.1 Run Demo Inference ===
print("=" * 60)
print("STEP 6: DEMO INFERENCE")
print("=" * 60)

run("python src/demo_inference.py --config configs/config.yaml")

# %%
# === 8.2 Display Demo Images ===
demo_files = sorted(glob.glob("outputs/demo/*.png") + glob.glob("outputs/demo/*.jpg"))
for demo_file in demo_files:
    print(f"\n[Demo Image] {os.path.basename(demo_file)}")
    display(Image(filename=demo_file, width=900))

# %% [markdown]
# ## 9. Summary & Model Comparison

# %%
# === 9.1 Final Summary ===
print("=" * 60)
print("PIPELINE COMPLETE - FINAL SUMMARY")
print("=" * 60)

# Model file sizes
print("\nModel File Sizes:")
for name, path in [
    ("ONNX (FP32)", "outputs/yolov5n.onnx"), 
    ("TRT FP32", "outputs/yolov5n_fp32.engine"),
    ("TRT FP16", "outputs/yolov5n_fp16.engine"),
    ("TRT INT8", "outputs/yolov5n_int8.engine"),
]:
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  {name:<15}: {size_mb:.1f} MB")

# Combined results table
print("\nCombined Results:")
print("-" * 90)
print(f"{'Model':<20} {'Latency (ms)':<15} {'FPS':<10} {'mAP@0.5':<12} {'mAP@0.5:0.95':<15} {'Size (MB)':<10}")
print("-" * 90)

# Helper for robust model lookup across different JSON key formats
def _find_model_data(data_dict, model_name):
    if not data_dict or not isinstance(data_dict, dict):
        return {}
    if model_name in data_dict:
        return data_dict[model_name]
    
    target = model_name.lower().replace("_", " ").replace("tensorrt", "trt")
    for k, v in data_dict.items():
        k_clean = k.lower().replace("_", " ").replace("tensorrt", "trt")
        if k_clean == target or set(k_clean.split()) == set(target.split()):
            return v
        if "pytorch" in target and "pytorch" in k_clean:
            return v
    return {}

# Load benchmark & accuracy JSON files directly from disk
bench_results_file = "outputs/results/benchmark_latency.json"
acc_results_file = "outputs/results/accuracy_results.json"

bench_data = {}
if os.path.exists(bench_results_file):
    with open(bench_results_file) as f:
        bench_data = json.load(f)

acc_data = {}
if os.path.exists(acc_results_file):
    with open(acc_results_file) as f:
        acc_data = json.load(f)

model_variants = ["PyTorch FP32", "TRT FP32", "TRT FP16", "TRT INT8"]
engine_paths = {
    "PyTorch FP32": None,
    "TRT FP32": "outputs/yolov5n_fp32.engine",
    "TRT FP16": "outputs/yolov5n_fp16.engine",
    "TRT INT8": "outputs/yolov5n_int8.engine",
}

for model_name in model_variants:
    latency = fps = map50 = map50_95 = size_mb = "N/A"

    bench = _find_model_data(bench_data, model_name)
    if bench:
        lat_val = bench.get('mean_ms', bench.get('mean_latency_ms', 0))
        fps_val = bench.get('fps', bench.get('throughput_fps', 0))
        if lat_val > 0:
            latency = f"{lat_val:.2f}"
        if fps_val > 0:
            fps = f"{fps_val:.1f}"

    acc = _find_model_data(acc_data, model_name)
    if acc:
        map50 = f"{acc.get('mAP50', 0):.4f}"
        map50_95 = f"{acc.get('mAP50-95', 0):.4f}"

    path = engine_paths.get(model_name)
    if path and os.path.exists(path):
        size_mb = f"{os.path.getsize(path) / (1024 * 1024):.1f}"

    print(f"  {model_name:<18} {latency:<15} {fps:<10} {map50:<12} {map50_95:<15} {size_mb:<10}")

print("-" * 90)

# Speedup summary
pt_bench = _find_model_data(bench_data, "PyTorch FP32")
int8_bench = _find_model_data(bench_data, "TRT INT8")
pytorch_lat = pt_bench.get("mean_ms", pt_bench.get("mean_latency_ms", 0))
int8_lat = int8_bench.get("mean_ms", int8_bench.get("mean_latency_ms", 0))
if pytorch_lat > 0 and int8_lat > 0:
    speedup = pytorch_lat / int8_lat
    print(f"\nINT8 Speedup over PyTorch FP32: {speedup:.2f}x")

print("\nPipeline completed successfully!")
print("All outputs saved in: outputs/")
print("Charts saved in: outputs/results/")
print("Demo images saved in: outputs/demo/")
