"""
Ablation phụ (Section 5 - ablation_plan.txt):
  5.1. Sweep số components K ∈ {100, 200, 300, 400} trên LFW (Baseline).
  5.2. So sánh cấu hình multi-scale: [80], [80, 40], [80, 40, 20] (accuracy + thời gian).

Usage: python run_ablation_extra.py
"""

import os
import sys
import time
import gc
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import functools
print = functools.partial(print, flush=True)

from src.config import (
    N_TRAIN_SAMPLES, RESULTS_DIR, FIGURES_DIR, TABLES_DIR,
    IMAGE_SIZE as CONFIG_IMAGE_SIZE,
)
from src.data_loader import load_casia_webface, load_all_eval_datasets
from src.eigenfaces import BaselineEigenfaces, EnhancedEigenfaces
from src.evaluation import evaluate_model

# Ablation dùng image_size=80 theo plan
ABLATION_IMAGE_SIZE = 80


def run_51_sweep_k(train_images, eval_datasets):
    """
    5.1. Baseline: K ∈ {100, 200, 300, 400}, LFW only, cosine.
    Mục tiêu: chứng minh K=300 là hợp lý.
    """
    print("\n" + "=" * 70)
    print("  5.1. ABLATION PHỤ: Số components K (Baseline, LFW, cosine)")
    print("=" * 70)

    lfw_images, lfw_issame = eval_datasets["LFW"]
    k_values = [100, 200, 300, 400]
    results = []

    for k in k_values:
        print(f"\n  --- K = {k} ---")
        t0 = time.time()
        model = BaselineEigenfaces(n_components=k, image_size=ABLATION_IMAGE_SIZE)
        model.fit(train_images, verbose=False)
        train_time = time.time() - t0

        res = evaluate_model(model, lfw_images, lfw_issame, "LFW", metric="cosine", verbose=True)
        acc = res["kfold_accuracy"] * 100
        std = res["kfold_std"] * 100
        infer_ms = res.get("avg_inference_ms", 0)

        results.append({
            "K": k,
            "Accuracy(%)": acc,
            "Std(%)": std,
            "AUC": res["auc"],
            "EER(%)": res["eer"] * 100,
            "Train_time_s": train_time,
            "Inference_ms": infer_ms,
        })
        print(f"  LFW: Acc = {acc:.2f}% ± {std:.2f}%, Train = {train_time:.1f}s, Infer = {infer_ms:.2f} ms")
        del model
        gc.collect()

    return results


def run_52_multiscale_configs(train_images, eval_datasets):
    """
    5.2. So sánh multi-scale: [80], [80, 40], [80, 40, 20].
    Mục tiêu: chứng minh [80, 40] cho trade-off tốt nhất.
    """
    print("\n" + "=" * 70)
    print("  5.2. ABLATION PHỤ: Cấu hình multi-scale (LFW, cosine)")
    print("=" * 70)

    lfw_images, lfw_issame = eval_datasets["LFW"]

    configs = [
        ("[80] (single)", [80], [300], False),
        ("[80, 40]", [80, 40], [300, 150], True),
        ("[80, 40, 20]", [80, 40, 20], [300, 150, 75], True),
    ]
    results = []

    for name, scales, n_comp, use_ms in configs:
        print(f"\n  --- {name} ---")
        t0 = time.time()
        if use_ms:
            model = EnhancedEigenfaces(
                scales=scales,
                n_components_per_scale=n_comp,
                use_illumination_norm=False,
                use_ensemble=False,
                use_multiscale=True,
            )
        else:
            model = BaselineEigenfaces(n_components=n_comp[0], image_size=scales[0])

        model.fit(train_images, verbose=False)
        train_time = time.time() - t0

        res = evaluate_model(model, lfw_images, lfw_issame, "LFW", metric="cosine", verbose=True)
        acc = res["kfold_accuracy"] * 100
        std = res["kfold_std"] * 100
        infer_ms = res.get("avg_inference_ms", 0)

        results.append({
            "Config": name,
            "Accuracy(%)": acc,
            "Std(%)": std,
            "AUC": res["auc"],
            "EER(%)": res["eer"] * 100,
            "Train_time_s": train_time,
            "Inference_ms": infer_ms,
        })
        print(f"  LFW: Acc = {acc:.2f}% ± {std:.2f}%, Train = {train_time:.1f}s, Infer = {infer_ms:.2f} ms")
        del model
        gc.collect()

    return results


def save_tables(results_51, results_52):
    """Lưu bảng kết quả vào results/tables/."""
    os.makedirs(TABLES_DIR, exist_ok=True)

    # 5.1
    path_51 = os.path.join(TABLES_DIR, "ablation_5_1_components_sweep.csv")
    with open(path_51, "w", encoding="utf-8") as f:
        f.write("K,Accuracy(%),Std(%),AUC,EER(%),Train_time_s,Inference_ms\n")
        for r in results_51:
            f.write(f"{r['K']},{r['Accuracy(%)']:.2f},{r['Std(%)']:.2f},"
                    f"{r['AUC']:.4f},{r['EER(%)']:.2f},{r['Train_time_s']:.1f},{r['Inference_ms']:.2f}\n")
    print(f"\n  Saved: {path_51}")

    # 5.2
    path_52 = os.path.join(TABLES_DIR, "ablation_5_2_multiscale_configs.csv")
    with open(path_52, "w", encoding="utf-8") as f:
        f.write("Config,Accuracy(%),Std(%),AUC,EER(%),Train_time_s,Inference_ms\n")
        for r in results_52:
            f.write(f"\"{r['Config']}\",{r['Accuracy(%)']:.2f},{r['Std(%)']:.2f},"
                    f"{r['AUC']:.4f},{r['EER(%)']:.2f},{r['Train_time_s']:.1f},{r['Inference_ms']:.2f}\n")
    print(f"  Saved: {path_52}")


def plot_and_save_figures(results_51, results_52):
    """Vẽ biểu đồ cột cho báo cáo/slide."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIGURES_DIR, exist_ok=True)

    # 5.1: Bar chart Accuracy vs K
    fig, ax = plt.subplots(figsize=(8, 5))
    ks = [r["K"] for r in results_51]
    accs = [r["Accuracy(%)"] for r in results_51]
    stds = [r["Std(%)"] for r in results_51]
    bars = ax.bar([str(k) for k in ks], accs, yerr=stds, capsize=5, color="steelblue", edgecolor="navy")
    ax.set_xlabel("Số components K")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("5.1. Ablation: Baseline LFW Accuracy theo K (cosine)")
    ax.set_ylim(0, 100)
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{a:.1f}%", ha="center", fontsize=10)
    plt.tight_layout()
    path1 = os.path.join(FIGURES_DIR, "ablation_5_1_components_sweep.png")
    plt.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure: {path1}")

    # 5.2: Bar chart Accuracy vs Config
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [r["Config"] for r in results_52]
    accs = [r["Accuracy(%)"] for r in results_52]
    stds = [r["Std(%)"] for r in results_52]
    x = range(len(labels))
    bars = ax.bar(x, accs, yerr=stds, capsize=5, color="coral", edgecolor="darkred")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("5.2. Ablation: Multi-scale configs trên LFW (cosine)")
    ax.set_ylim(0, 100)
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{a:.1f}%", ha="center", fontsize=10)
    plt.tight_layout()
    path2 = os.path.join(FIGURES_DIR, "ablation_5_2_multiscale_configs.png")
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure: {path2}")


def print_summary(results_51, results_52):
    """In bảng tổng hợp ra console."""
    print("\n" + "=" * 70)
    print("  BẢNG 5.1. Sweep K (Baseline, LFW, cosine)")
    print("=" * 70)
    print(f"  {'K':<6} {'Accuracy(%)':<12} {'Std(%)':<8} {'AUC':<8} {'Train(s)':<10} {'Infer(ms)':<10}")
    print("  " + "-" * 56)
    for r in results_51:
        print(f"  {r['K']:<6} {r['Accuracy(%)']:<12.2f} {r['Std(%)']:<8.2f} "
              f"{r['AUC']:<8.4f} {r['Train_time_s']:<10.1f} {r['Inference_ms']:<10.2f}")

    print("\n" + "=" * 70)
    print("  BẢNG 5.2. Cấu hình multi-scale (LFW, cosine)")
    print("=" * 70)
    print(f"  {'Config':<20} {'Accuracy(%)':<12} {'Std(%)':<8} {'Train(s)':<10} {'Infer(ms)':<10}")
    print("  " + "-" * 62)
    for r in results_52:
        print(f"  {r['Config']:<20} {r['Accuracy(%)']:<12.2f} {r['Std(%)']:<8.2f} "
              f"{r['Train_time_s']:<10.1f} {r['Inference_ms']:<10.2f}")


def main():
    print("\n" + "=" * 70)
    print("  ABLATION PHỤ (Section 5 - ablation_plan.txt)")
    print("  5.1. Sweep K ∈ {100, 200, 300, 400}")
    print("  5.2. Multi-scale: [80], [80, 40], [80, 40, 20]")
    print("=" * 70)
    print(f"  Image size: {ABLATION_IMAGE_SIZE}x{ABLATION_IMAGE_SIZE}")
    print(f"  N_TRAIN_SAMPLES: {N_TRAIN_SAMPLES}")

    t0 = time.time()
    print("\n  Loading data...")
    train_images, _ = load_casia_webface(
        n_samples=N_TRAIN_SAMPLES,
        image_size=ABLATION_IMAGE_SIZE,
    )
    eval_datasets = load_all_eval_datasets(image_size=ABLATION_IMAGE_SIZE)
    print(f"  Done ({time.time() - t0:.1f}s). Train: {train_images.shape}, LFW pairs: {len(eval_datasets['LFW'][1])}")

    results_51 = run_51_sweep_k(train_images, eval_datasets)
    results_52 = run_52_multiscale_configs(train_images, eval_datasets)

    print_summary(results_51, results_52)
    save_tables(results_51, results_52)
    plot_and_save_figures(results_51, results_52)

    print("\n" + "=" * 70)
    print("  Ablation phụ hoàn tất.")
    print("  Tables: results/tables/ablation_5_1_*.csv, ablation_5_2_*.csv")
    print("  Figures: results/figures/ablation_5_1_*.png, ablation_5_2_*.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
