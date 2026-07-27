"""
Script to visualize model comparison results.
Generates grouped bar charts for Performance Benchmark and Accuracy Comparison.
"""

import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.common import setup_logger, load_config

logger = setup_logger("visualize_results")


def set_dark_theme():
    """Apply a professional dark theme to matplotlib."""
    plt.style.use('dark_background')
    plt.rcParams.update({
        "axes.facecolor": "#1c1c1e",
        "figure.facecolor": "#121212",
        "axes.edgecolor": "#333333",
        "grid.color": "#333333",
        "text.color": "#ffffff",
        "axes.labelcolor": "#cccccc",
        "xtick.color": "#cccccc",
        "ytick.color": "#cccccc",
        "font.family": "sans-serif",
    })


def plot_performance(results_path, output_dir):
    """Plot performance benchmark (Latency and Throughput)."""
    if not os.path.exists(results_path):
        logger.warning(f"Results file not found: {results_path}")
        return

    with open(results_path, "r") as f:
        data = json.load(f)

    models = []
    latencies = []
    latency_stds = []
    fps_list = []

    for model, metrics in data.items():
        models.append(model)
        latencies.append(metrics.get("mean_ms", metrics.get("mean_latency_ms", 0)))
        latency_stds.append(metrics.get("std_ms", metrics.get("std_latency_ms", 0)))
        fps_list.append(metrics.get("fps", metrics.get("throughput_fps", 0)))

    x = np.arange(len(models))
    width = 0.6

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Gradient-like blues/greens colors for bars
    colors = ["#1f77b4", "#2ca02c", "#17becf", "#9467bd"][:len(models)]
    
    bars = ax1.bar(x, latencies, width, yerr=latency_stds, capsize=5,
                   color=colors, alpha=0.85, edgecolor="white", label="Latency (ms)")

    ax1.set_xlabel("Model Variants", fontsize=12, labelpad=10)
    ax1.set_ylabel("Latency (ms)", fontsize=12)
    ax1.set_title("YOLOv5n Inference Performance: FP32 vs FP16 vs INT8", fontsize=14, pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=11)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    # Dual axis for FPS
    ax2 = ax1.twinx()
    ax2.plot(x, fps_list, color="#ff7f0e", marker="D", markersize=8, linewidth=2, label="Throughput (FPS)")
    ax2.set_ylabel("Throughput (FPS)", fontsize=12)

    # Legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left", frameon=False)

    # Annotate bars
    for bar in bars:
        yval = bar.get_height()
        if yval > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.1f}", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    out_file = os.path.join(output_dir, "performance_benchmark.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved performance chart to {out_file}")


def plot_accuracy(results_path, output_dir):
    """Plot accuracy comparison (mAP@0.5 and mAP@0.5:0.95)."""
    if not os.path.exists(results_path):
        logger.warning(f"Results file not found: {results_path}")
        return

    with open(results_path, "r") as f:
        data = json.load(f)

    models = list(data.keys())
    map50 = [data[m].get("mAP50", 0) for m in models]
    map50_95 = [data[m].get("mAP50-95", 0) for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    rects1 = ax.bar(x - width/2, map50, width, color="#f39c12", label="mAP@0.5")
    rects2 = ax.bar(x + width/2, map50_95, width, color="#e74c3c", label="mAP@0.5:0.95")

    ax.set_ylabel("mAP Score", fontsize=12)
    ax.set_title("YOLOv5n Accuracy: FP32 vs FP16 vs INT8", fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(frameon=False)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f"{height:.3f}",
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    out_file = os.path.join(output_dir, "accuracy_comparison.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved accuracy chart to {out_file}")


def main():
    set_dark_theme()
    config = load_config()
    
    results_dir = config["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    perf_path = os.path.join(results_dir, "benchmark_latency.json")
    acc_path = os.path.join(results_dir, "accuracy_results.json")

    plot_performance(perf_path, results_dir)
    plot_accuracy(acc_path, results_dir)


if __name__ == "__main__":
    main()
