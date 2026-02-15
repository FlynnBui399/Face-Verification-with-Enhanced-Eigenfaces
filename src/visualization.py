"""
Visualization - Vẽ biểu đồ và lưu kết quả.

Các visualizations:
  1. Eigenfaces gallery
  2. ROC curves
  3. Ablation study bar chart
  4. Image reconstruction comparison
  5. Distance distribution histograms
  6. Explained variance plot

Author: Mathematics for AI - Final Project
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from src.config import FIGURES_DIR


def ensure_fig_dir():
    os.makedirs(FIGURES_DIR, exist_ok=True)


# ==============================================================================
# 1. EIGENFACES VISUALIZATION
# ==============================================================================

def plot_eigenfaces(eigenfaces, n_show=16, title="Top Eigenfaces", 
                    filename="eigenfaces.png"):
    """
    Visualize top eigenfaces (principal components) dưới dạng ảnh.
    
    Args:
        eigenfaces: np.ndarray shape (K, H, W) - eigenfaces
        n_show: Số eigenfaces hiển thị
        title: Title of the plot
        filename: Save filename
    """
    ensure_fig_dir()
    
    n_show = min(n_show, len(eigenfaces))
    cols = 4
    rows = (n_show + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    for i in range(rows * cols):
        ax = axes[i // cols][i % cols] if rows > 1 else axes[i % cols]
        if i < n_show:
            face = eigenfaces[i]
            # Normalize for display
            face_norm = (face - face.min()) / (face.max() - face.min() + 1e-10)
            ax.imshow(face_norm, cmap='gray')
            ax.set_title(f"PC {i+1}", fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    filepath = os.path.join(FIGURES_DIR, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {filepath}")


def plot_mean_face(mean_face, title="Mean Face", filename="mean_face.png"):
    """Visualize mean face."""
    ensure_fig_dir()
    
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    face_norm = (mean_face - mean_face.min()) / (mean_face.max() - mean_face.min() + 1e-10)
    ax.imshow(face_norm, cmap='gray')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    filepath = os.path.join(FIGURES_DIR, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {filepath}")


# ==============================================================================
# 2. ROC CURVES
# ==============================================================================

def plot_roc_curves(results_dict, dataset_name="LFW", filename="roc_curves.png"):
    """
    Plot ROC curves cho multiple methods/metrics.
    
    Args:
        results_dict: {method_name: evaluation_results}
        dataset_name: For title
        filename: Save filename
    """
    ensure_fig_dir()
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0', '#00BCD4']
    
    for i, (name, results) in enumerate(results_dict.items()):
        if 'roc' not in results:
            continue
        roc = results['roc']
        color = colors[i % len(colors)]
        auc = roc['auc']
        
        # Sort by FPR for smooth curve
        sorted_idx = np.argsort(roc['fpr'])
        ax.plot(roc['fpr'][sorted_idx], roc['tpr'][sorted_idx],
                color=color, linewidth=2, label=f"{name} (AUC={auc:.4f})")
    
    # Diagonal (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label="Random")
    
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=12)
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=12)
    ax.set_title(f'ROC Curves - {dataset_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    filepath = os.path.join(FIGURES_DIR, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {filepath}")


# ==============================================================================
# 3. ABLATION STUDY BAR CHART
# ==============================================================================

def plot_ablation_study(ablation_results, filename="ablation_study.png"):
    """
    Ablation study bar chart.
    
    Args:
        ablation_results: dict {method_name: {dataset_name: accuracy}}
    """
    ensure_fig_dir()
    
    methods = list(ablation_results.keys())
    datasets = list(next(iter(ablation_results.values())).keys())
    
    n_methods = len(methods)
    n_datasets = len(datasets)
    
    x = np.arange(n_methods)
    width = 0.8 / n_datasets
    
    colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800']
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    for i, dataset in enumerate(datasets):
        accs = [ablation_results[m][dataset] * 100 for m in methods]
        bars = ax.bar(x + i * width - (n_datasets-1)*width/2, accs, width,
                      label=dataset, color=colors[i % len(colors)], alpha=0.85)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Ablation Study - Contribution of Each Improvement', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha='right', fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 105])
    
    filepath = os.path.join(FIGURES_DIR, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {filepath}")


# ==============================================================================
# 4. DISTANCE DISTRIBUTION
# ==============================================================================

def plot_distance_distribution(distances, issame, metric_name="Euclidean",
                                dataset_name="LFW", 
                                filename="distance_distribution.png"):
    """
    Histogram phân phối khoảng cách cho positive/negative pairs.
    """
    ensure_fig_dir()
    
    pos_distances = distances[issame]
    neg_distances = distances[~issame]
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    
    ax.hist(pos_distances, bins=50, alpha=0.6, color='#4CAF50', 
            label=f'Same Person (n={len(pos_distances)})', density=True)
    ax.hist(neg_distances, bins=50, alpha=0.6, color='#F44336',
            label=f'Different Person (n={len(neg_distances)})', density=True)
    
    ax.set_xlabel(f'{metric_name} Distance', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Distance Distribution - {dataset_name} ({metric_name})',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    filepath = os.path.join(FIGURES_DIR, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {filepath}")


# ==============================================================================
# 5. EXPLAINED VARIANCE
# ==============================================================================

def plot_explained_variance(eigenvalues, filename="explained_variance.png"):
    """
    Plot explained variance ratio và cumulative explained variance.
    """
    ensure_fig_dir()
    
    total = np.sum(eigenvalues)
    ratios = eigenvalues / total if total > 0 else eigenvalues
    cumulative = np.cumsum(ratios)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Individual explained variance
    n_show = min(50, len(ratios))
    ax1.bar(range(1, n_show+1), ratios[:n_show] * 100, color='#2196F3', alpha=0.7)
    ax1.set_xlabel('Principal Component', fontsize=12)
    ax1.set_ylabel('Explained Variance (%)', fontsize=12)
    ax1.set_title('Individual Explained Variance Ratio', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Cumulative
    ax2.plot(range(1, len(cumulative)+1), cumulative * 100, 
             color='#F44336', linewidth=2)
    ax2.axhline(y=95, color='gray', linestyle='--', alpha=0.5, label='95%')
    ax2.axhline(y=99, color='gray', linestyle=':', alpha=0.5, label='99%')
    
    # Find n_components for 95% and 99%
    n_95 = np.argmax(cumulative >= 0.95) + 1 if np.any(cumulative >= 0.95) else len(cumulative)
    n_99 = np.argmax(cumulative >= 0.99) + 1 if np.any(cumulative >= 0.99) else len(cumulative)
    ax2.axvline(x=n_95, color='#4CAF50', linestyle='--', alpha=0.7, 
                label=f'95% at n={n_95}')
    ax2.axvline(x=n_99, color='#FF9800', linestyle='--', alpha=0.7,
                label=f'99% at n={n_99}')
    
    ax2.set_xlabel('Number of Components', fontsize=12)
    ax2.set_ylabel('Cumulative Explained Variance (%)', fontsize=12)
    ax2.set_title('Cumulative Explained Variance', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(FIGURES_DIR, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {filepath}")


# ==============================================================================
# 6. RECONSTRUCTION COMPARISON
# ==============================================================================

def plot_reconstruction(original, reconstructions, n_components_list,
                        filename="reconstruction.png"):
    """
    So sánh ảnh gốc vs ảnh reconstructed với số components khác nhau.
    
    Args:
        original: np.ndarray shape (n_show, H, W)
        reconstructions: dict {n_components: np.ndarray shape (n_show, H, W)}
        n_components_list: List số components
    """
    ensure_fig_dir()
    
    n_show = min(5, len(original))
    n_cols = 1 + len(n_components_list)
    
    fig, axes = plt.subplots(n_show, n_cols, figsize=(3*n_cols, 3*n_show))
    
    # Column headers
    titles = ["Original"] + [f"K={k}" for k in n_components_list]
    
    for j, title in enumerate(titles):
        if n_show > 1:
            axes[0][j].set_title(title, fontsize=11, fontweight='bold')
        else:
            axes[j].set_title(title, fontsize=11, fontweight='bold')
    
    for i in range(n_show):
        # Original
        ax = axes[i][0] if n_show > 1 else axes[0]
        img = original[i]
        img_norm = (img - img.min()) / (img.max() - img.min() + 1e-10)
        ax.imshow(img_norm, cmap='gray')
        ax.axis('off')
        
        # Reconstructions
        for j, k in enumerate(n_components_list):
            ax = axes[i][j+1] if n_show > 1 else axes[j+1]
            img = reconstructions[k][i]
            img_norm = (img - img.min()) / (img.max() - img.min() + 1e-10)
            ax.imshow(img_norm, cmap='gray')
            ax.axis('off')
    
    plt.suptitle('Image Reconstruction with Different Number of Components',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    filepath = os.path.join(FIGURES_DIR, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {filepath}")


# ==============================================================================
# 7. SAVE RESULTS TABLE TO CSV
# ==============================================================================

def save_results_table(results, filename="results_table.csv"):
    """Save results dictionary to CSV."""
    from src.config import TABLES_DIR
    os.makedirs(TABLES_DIR, exist_ok=True)
    
    filepath = os.path.join(TABLES_DIR, filename)
    
    with open(filepath, 'w') as f:
        # Header
        f.write("Method,Dataset,Metric,Accuracy(%),Std(%),AUC,EER(%),Inference_ms\n")
        
        for method_name, datasets in results.items():
            for dataset_name, r in datasets.items():
                f.write(f"{method_name},{dataset_name},{r.get('metric','euc')},"
                        f"{r['kfold_accuracy']*100:.2f},{r['kfold_std']*100:.2f},"
                        f"{r['auc']:.4f},{r['eer']*100:.2f},"
                        f"{r.get('avg_inference_ms', 0):.2f}\n")
    
    print(f"  [Saved] {filepath}")
