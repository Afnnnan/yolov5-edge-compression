# YOLOv5n Model Compression Pipeline for Edge Deployment

A complete, repeatable pipeline for compressing **YOLOv5-nano** via ONNX export and TensorRT INT8 post-training quantization (PTQ), benchmarked on an NVIDIA T4 GPU.

## Overview

This project demonstrates how to take a pre-trained object detection model and optimize it for edge deployment through:

1. **ONNX Export**: Convert PyTorch model to ONNX format
2. **TensorRT Optimization**: Build optimized engines at FP32, FP16, and INT8 precision
3. **INT8 Post-Training Quantization**: Calibrate activation ranges using representative data
4. **Benchmarking**: Measure latency, throughput (FPS), and accuracy (mAP)
5. **Visualization**: Generate comparison charts and demo inference images

## Project Structure

```
topcoder/
├── src/                          # Pipeline scripts
│   ├── export_onnx.py            # Step 1: PyTorch -> ONNX
│   ├── build_engine.py           # Step 2: ONNX -> TensorRT engines
│   ├── benchmark_latency.py      # Step 3: Latency & throughput
│   ├── evaluate_accuracy.py      # Step 4: mAP on COCO val2017
│   ├── visualize_results.py      # Step 5: Comparison charts
│   └── demo_inference.py         # Step 6: Visual demo
├── utils/                        # Utility modules
│   ├── common.py                 # Config, logging, timing
│   ├── coco_utils.py             # COCO dataset & preprocessing
│   ├── calibrator.py             # TensorRT INT8 calibrator
│   └── trt_inference.py          # TensorRT inference wrapper
├── configs/
│   └── config.yaml               # Pipeline configuration
├── notebooks/
│   └── kaggle_pipeline.py        # Kaggle notebook script
├── requirements.txt
└── README.md
```

## Quick Start (Kaggle)

### Prerequisites
- Kaggle account with GPU (T4) access
- Internet-enabled notebook

### Steps

1. **Push this repo to GitHub**
2. **Create a new Kaggle notebook** with GPU accelerator enabled
3. **Copy cells from** `notebooks/kaggle_pipeline.py` into the notebook
4. **Update the `REPO_URL`** in cell 1.2 to point to your GitHub repo
5. **Run all cells**, which will:
   - Install TensorRT and dependencies
   - Clone the repo
   - Download COCO val2017 (~1GB)
   - Run the full compression pipeline
   - Generate benchmark results and charts

### Expected Runtime
| Step | Time |
|------|------|
| Setup & Data Download | ~5 min |
| ONNX Export | ~30 sec |
| TRT Engine Build (FP32+FP16) | ~3 min |
| TRT Engine Build (INT8 + calibration) | ~5 min |
| Latency Benchmarking | ~2 min |
| mAP Evaluation (1000 images) | ~15 min |
| Visualization & Demo | ~2 min |
| **Total** | **~30 min** |

## Expected Results (T4 GPU)

> These are approximate values. Exact numbers will vary.

| Model | Latency (ms) | FPS | mAP@0.5 | Size (MB) |
|-------|-------------|-----|---------|-----------|
| PyTorch FP32 | ~8-10 | ~100-120 | ~0.46 | ~7.2 |
| TRT FP32 | ~4-6 | ~170-250 | ~0.46 | ~12 |
| TRT FP16 | ~1.5-2.5 | ~400-600 | ~0.46 | ~5 |
| TRT INT8 | ~1.0-1.5 | ~650-1000 | ~0.44-0.46 | ~4 |

## Running Individual Steps

Each script can be run independently from the project root:

```bash
# Step 1: Export ONNX
python src/export_onnx.py --config configs/config.yaml

# Step 2: Build TensorRT engines
python src/build_engine.py --precision fp32 fp16 int8

# Step 3: Benchmark latency
python src/benchmark_latency.py

# Step 4: Evaluate accuracy
python src/evaluate_accuracy.py --num-images 1000

# Step 5: Generate charts
python src/visualize_results.py

# Step 6: Demo inference
python src/demo_inference.py
```

## Technical Details

### Model
- **YOLOv5n-u** (ultralytics-native variant, ~3.2M parameters)
- Pre-trained on COCO (80 classes)
- Input: 640x640 RGB

### INT8 Quantization
- **Algorithm**: Entropy Calibration v2 (IInt8EntropyCalibrator2)
- **Calibration set**: 500 random COCO val2017 images
- **Method**: Post-Training Quantization (PTQ), no retraining required

### Evaluation
- **Dataset**: COCO val2017 (5000 images)
- **Metrics**: mAP@0.5, mAP@0.5:0.95 (via pycocotools)
- **Benchmarking**: CUDA events for GPU timing, 50 warmup + 300 measurement iterations

### Key Dependencies
- PyTorch >= 2.0
- Ultralytics >= 8.3
- TensorRT >= 10.0 (auto-installed on Kaggle)
- ONNX >= 1.14
- pycocotools >= 2.0.6

## Notes

- **Platform-specific engines**: TensorRT engines built on T4 are optimized for T4 architecture only. They would need to be rebuilt on Jetson for deployment.
- **Benchmark context**: Results are from a cloud T4 GPU, not a Jetson device. The pipeline and methodology are identical; only the absolute performance numbers differ.
- **INT8 accuracy**: Typical mAP drop with PTQ is 0.1-0.5 points. If the drop is larger, increase calibration images or try MinMax calibration.

## License

This project is created for the Topcoder challenge.
