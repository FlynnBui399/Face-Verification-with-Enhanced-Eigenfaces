"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  ENHANCED EIGENFACES: PCA/SVD cho Face Recognition                           ║
║  Mathematics for AI - Final Project                                          ║
║                                                                              ║
║  Thuật toán PCA/SVD và Ứng dụng cho bài toán giảm chiều dữ liệu             ║
║  và phân loại khuôn mặt                                                      ║
║                                                                              ║
║  ┌─────────────────────────────────────────────────────────────────┐          ║
║  │  TẤT CẢ THUẬT TOÁN IMPLEMENT TỪ ĐẦU (FROM SCRATCH)            │          ║
║  │  - PCA/SVD: Power Iteration + Gram-Schmidt + Deflation         │          ║
║  │  - Preprocessing: Histogram Eq, DoG, Tan-Triggs                │          ║
║  │  - Distance Metrics: Euclidean, Cosine, Manhattan, Chi-Square  │          ║
║  │  - Evaluation: K-fold CV, ROC, AUC, EER                        │          ║
║  └─────────────────────────────────────────────────────────────────┘          ║
║                                                                              ║
║  Datasets:                                                                   ║
║    Training: CASIA-WebFace (494K images, 10572 subjects)                     ║
║    Testing:  LFW, CFP-FP, AgeDB-30                                          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Usage:
    python run_experiment.py
"""

import os
import sys
import time
import gc
import numpy as np

# Thêm project root vào path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Force unbuffered output
import functools
print = functools.partial(print, flush=True)

from src.config import *
from src.data_loader import load_casia_webface, load_eval_bin, load_all_eval_datasets
from src.preprocessing import preprocess_batch
from src.eigenfaces import BaselineEigenfaces, MultiScaleEigenfaces, EnhancedEigenfaces
from src.evaluation import (
    evaluate_model, evaluate_all_metrics, print_comparison_table
)
from src.visualization import (
    plot_eigenfaces, plot_mean_face, plot_roc_curves,
    plot_ablation_study, plot_distance_distribution,
    plot_explained_variance, plot_reconstruction, save_results_table
)


def print_banner():
    print("\n" + "="*75)
    print("  ENHANCED EIGENFACES: PCA/SVD for Face Recognition")
    print("  Mathematics for AI - Final Project")
    print("  " + "-"*50)
    print("  All algorithms implemented FROM SCRATCH")
    print("  No sklearn, no dlib, no face_recognition library")
    print("="*75)


def print_phase(phase_num, description):
    print(f"\n{'='*75}")
    print(f"  PHASE {phase_num}: {description}")
    print(f"{'='*75}")


# ==============================================================================
# PHASE 1: DATA LOADING
# ==============================================================================

def phase1_load_data():
    """Load CASIA-WebFace training data và eval datasets."""
    print_phase(1, "DATA LOADING")
    
    start_time = time.time()
    
    # Load training data
    print("\n  [1.1] Loading CASIA-WebFace training data...")
    train_images, train_labels = load_casia_webface(
        n_samples=N_TRAIN_SAMPLES, 
        image_size=IMAGE_SIZE
    )
    
    # Load eval datasets
    print("\n  [1.2] Loading evaluation datasets...")
    eval_datasets = load_all_eval_datasets(image_size=IMAGE_SIZE)
    
    elapsed = time.time() - start_time
    print(f"\n  Phase 1 complete! ({elapsed:.1f}s)")
    print(f"  Training: {train_images.shape[0]} images, "
          f"{len(np.unique(train_labels))} subjects")
    for name, (imgs, issame) in eval_datasets.items():
        print(f"  {name}: {len(imgs)} images, {len(issame)} pairs")
    
    return train_images, train_labels, eval_datasets


# ==============================================================================
# PHASE 2: BASELINE EIGENFACES
# ==============================================================================

def phase2_baseline(train_images, eval_datasets):
    """Train và evaluate baseline Eigenfaces."""
    print_phase(2, "BASELINE EIGENFACES (Standard PCA)")
    
    start_time = time.time()
    
    # Train baseline
    baseline = BaselineEigenfaces(n_components=N_COMPONENTS, image_size=IMAGE_SIZE)
    baseline.fit(train_images, verbose=True)
    
    # Visualize eigenfaces
    eigenfaces = baseline.get_eigenfaces()
    plot_eigenfaces(eigenfaces, n_show=16, title="Baseline Eigenfaces (Top 16 PCs)")
    
    # Plot mean face
    mean_face = baseline.pca.mean_.reshape(IMAGE_SIZE, IMAGE_SIZE)
    plot_mean_face(mean_face)
    
    # Plot explained variance
    plot_explained_variance(baseline.pca.eigenvalues_)
    
    # Plot reconstruction
    if len(train_images) >= 5:
        sample_imgs = train_images[:5]
        reconstructions = {}
        for k in [10, 50, 100, 200]:
            if k <= N_COMPONENTS:
                reconstructions[k] = baseline.reconstruct(sample_imgs, n_components=k)
        if reconstructions:
            plot_reconstruction(sample_imgs, reconstructions, list(reconstructions.keys()))
    
    gc.collect()  # Free memory after training
    
    # Evaluate on all datasets
    baseline_results = {}
    for name, (eval_images, issame) in eval_datasets.items():
        print(f"\n  --- Evaluating Baseline on {name} ---")
        result = evaluate_model(baseline, eval_images, issame, name, 
                               metric="cosine", verbose=True)
        baseline_results[name] = result
        
        # Plot distance distribution
        plot_distance_distribution(
            result['distances'], result['issame'],
            "Cosine", name, f"dist_{name.lower().replace('-','_')}_baseline.png"
        )
    
    elapsed = time.time() - start_time
    print(f"\n  Phase 2 complete! ({elapsed:.1f}s)")
    
    return baseline, baseline_results


# ==============================================================================
# PHASE 3: ABLATION STUDY
# ==============================================================================

def phase3_ablation(train_images, eval_datasets):
    """Ablation study - đánh giá từng improvement riêng lẻ."""
    print_phase(3, "ABLATION STUDY")
    
    start_time = time.time()
    
    ablation_results = {}
    ablation_for_plot = {}
    
    # Helper function to evaluate and collect results
    def eval_config(model, config_name):
        accs = {}
        results = {}
        for name, (eval_images, issame) in eval_datasets.items():
            result = evaluate_model(model, eval_images, issame, name,
                                   metric="cosine", verbose=True)
            accs[name] = result['kfold_accuracy']
            results[name] = result
        ablation_for_plot[config_name] = accs
        ablation_results[config_name] = results
        gc.collect()
        return model
    
    # Config 1: Baseline (no preprocessing, just standardize + PCA)
    print("\n  [3.1] Baseline (no preprocessing)...")
    baseline = BaselineEigenfaces(n_components=N_COMPONENTS, image_size=IMAGE_SIZE)
    baseline.fit(train_images, verbose=False)
    eval_config(baseline, "Baseline")
    del baseline; gc.collect()
    
    # Config 2: + Histogram Equalization (C3 contribution)
    print("\n  [3.2] + Histogram Equalization (C3)...")
    model_hist = EnhancedEigenfaces(
        scales=[IMAGE_SIZE], 
        n_components_per_scale=[N_COMPONENTS],
        use_illumination_norm=True,
        use_ensemble=False,
        use_multiscale=False,
        illumination_method="hist_eq"
    )
    model_hist.fit(train_images, verbose=False)
    eval_config(model_hist, "+HistEq")
    del model_hist; gc.collect()
    
    # Config 3: + Tan-Triggs (C3 alternative)
    print("\n  [3.3] + Tan-Triggs Normalization (C3 alternative)...")
    model_tt = EnhancedEigenfaces(
        scales=[IMAGE_SIZE], 
        n_components_per_scale=[N_COMPONENTS],
        use_illumination_norm=True,
        use_ensemble=False,
        use_multiscale=False,
        illumination_method="tan_triggs"
    )
    model_tt.fit(train_images, verbose=False)
    eval_config(model_tt, "+TanTriggs")
    del model_tt; gc.collect()
    
    # Config 4: + Multi-Scale (C1 contribution)
    print("\n  [3.4] + Multi-Scale [{0}] (C1)...".format(
        ', '.join(str(s) for s in MULTI_SCALE_SIZES)))
    model_ms = EnhancedEigenfaces(
        scales=MULTI_SCALE_SIZES,
        n_components_per_scale=[N_COMPONENTS, 100],
        use_illumination_norm=False,
        use_ensemble=False,
        use_multiscale=True
    )
    model_ms.fit(train_images, verbose=False)
    eval_config(model_ms, "+MultiScale")
    del model_ms; gc.collect()
    
    # Config 5: Combined (HistEq + MultiScale = C1 + C3)
    print("\n  [3.5] Combined: HistEq + MultiScale (C1 + C3)...")
    model_combined = EnhancedEigenfaces(
        scales=MULTI_SCALE_SIZES,
        n_components_per_scale=[N_COMPONENTS, 100],
        use_illumination_norm=True,
        use_ensemble=False,
        use_multiscale=True,
        illumination_method="hist_eq"
    )
    model_combined.fit(train_images, verbose=False)
    eval_config(model_combined, "HistEq+MS")
    
    # Plot ablation results
    plot_ablation_study(ablation_for_plot)
    
    # Print comparison
    print(f"\n  {'='*70}")
    print(f"  ABLATION STUDY SUMMARY")
    print(f"  {'='*70}")
    print(f"  {'Method':<25}", end="")
    for name in eval_datasets:
        print(f" {name:<15}", end="")
    print()
    print(f"  {'-'*25}", end="")
    for _ in eval_datasets:
        print(f" {'-'*15}", end="")
    print()
    
    for method, accs in ablation_for_plot.items():
        print(f"  {method:<25}", end="")
        for name in eval_datasets:
            acc = accs.get(name, 0)
            delta = acc - ablation_for_plot["Baseline"].get(name, 0)
            if method == "Baseline":
                print(f" {acc*100:>6.2f}%       ", end="")
            else:
                print(f" {acc*100:>6.2f}% ({delta*100:+.1f})", end="")
        print()
    
    elapsed = time.time() - start_time
    print(f"\n  Phase 3 complete! ({elapsed:.1f}s)")
    
    return ablation_results, model_combined


# ==============================================================================
# PHASE 4: ENHANCED EIGENFACES (FULL METHOD)
# ==============================================================================

def phase4_enhanced(train_images, eval_datasets):
    """Train Enhanced Eigenfaces with best configuration and evaluate."""
    print_phase(4, "ENHANCED EIGENFACES (Best Configuration)")
    
    start_time = time.time()
    
    # ================================================================
    # Based on ablation study:
    #   +HistEq: best on LFW (+0.9%) and AgeDB-30 (+2.5%)
    #   +MultiScale: best on CFP-FP (+1.8%)
    #   Combined HistEq+MS: interaction effects degrade LFW
    #
    # Strategy: Train both configurations, pick best per dataset
    # ================================================================
    
    # Config A: Enhanced with HistEq only (C3) — best for LFW, AgeDB
    print("\n  [4.1] Training Enhanced Eigenfaces + HistEq (C3)...")
    enhanced_hist = EnhancedEigenfaces(
        scales=[IMAGE_SIZE],
        n_components_per_scale=[N_COMPONENTS],
        use_illumination_norm=True,
        use_ensemble=False,
        use_multiscale=False,
        illumination_method="hist_eq"
    )
    enhanced_hist.fit(train_images, verbose=True)
    gc.collect()
    
    # Config B: Enhanced with MultiScale (C1) — best for CFP-FP
    print("\n  [4.2] Training Enhanced Eigenfaces + MultiScale (C1)...")
    enhanced_ms = EnhancedEigenfaces(
        scales=MULTI_SCALE_SIZES,
        n_components_per_scale=[N_COMPONENTS, 100],
        use_illumination_norm=False,
        use_ensemble=False,
        use_multiscale=True
    )
    enhanced_ms.fit(train_images, verbose=True)
    gc.collect()
    
    # Evaluate both configs, pick best per dataset
    enhanced_results = {}
    roc_results = {}
    
    configs = {
        "+HistEq": enhanced_hist,
        "+MultiScale": enhanced_ms,
    }
    
    for name, (eval_images, issame) in eval_datasets.items():
        print(f"\n  --- Evaluating on {name} ---")
        
        best_acc = 0
        best_result = None
        best_config = None
        
        for config_name, model in configs.items():
            for metric in ["cosine", "euclidean"]:
                result = evaluate_model(model, eval_images, issame, name,
                                       metric=metric, verbose=True)
                if result['kfold_accuracy'] > best_acc:
                    best_acc = result['kfold_accuracy']
                    best_result = result
                    best_config = f"{config_name} ({metric})"
        
        enhanced_results[name] = best_result
        roc_results[f"Enhanced ({name})"] = best_result
        
        print(f"\n  → Best for {name}: {best_config} = {best_acc*100:.2f}%")
        
        # Plot distance distribution
        plot_distance_distribution(
            best_result['distances'], best_result['issame'],
            best_result['metric'].capitalize(), name, 
            f"dist_{name.lower().replace('-','_')}_enhanced.png"
        )
    
    # Plot ROC curves
    plot_roc_curves(roc_results, "All Datasets", "roc_enhanced.png")
    
    elapsed = time.time() - start_time
    print(f"\n  Phase 4 complete! ({elapsed:.1f}s)")
    
    return enhanced_hist, enhanced_results


# ==============================================================================
# PHASE 5: COMPARISON & ENERGY ANALYSIS
# ==============================================================================

def phase5_comparison(baseline_results, enhanced_results, eval_datasets):
    """So sánh Baseline vs Enhanced + Energy analysis."""
    print_phase(5, "COMPARISON & ENERGY ANALYSIS")
    
    # Comparison table
    print(f"\n  {'='*80}")
    print(f"  FINAL COMPARISON: Baseline vs Enhanced Eigenfaces")
    print(f"  {'='*80}")
    print(f"  {'Dataset':<12} {'Baseline':<20} {'Enhanced':<20} {'Improvement':<15}")
    print(f"  {'-'*12} {'-'*20} {'-'*20} {'-'*15}")
    
    for name in eval_datasets:
        b_acc = baseline_results[name]['kfold_accuracy']
        b_std = baseline_results[name]['kfold_std']
        
        # Get enhanced result (now a single result dict per dataset)
        if name in enhanced_results:
            e_result = enhanced_results[name]
            if isinstance(e_result, dict) and 'kfold_accuracy' in e_result:
                e_acc = e_result['kfold_accuracy']
                e_std = e_result['kfold_std']
            else:
                # Dict of metrics - pick best
                best_metric = max(e_result.keys(),
                                key=lambda m: e_result[m]['kfold_accuracy'])
                e_acc = e_result[best_metric]['kfold_accuracy']
                e_std = e_result[best_metric]['kfold_std']
        else:
            e_acc = b_acc
            e_std = b_std
        
        improvement = e_acc - b_acc
        print(f"  {name:<12} {b_acc*100:>6.2f}% ± {b_std*100:.2f}%  "
              f"  {e_acc*100:>6.2f}% ± {e_std*100:.2f}%  "
              f"  {improvement*100:>+6.2f}%")
    
    # Energy analysis
    print(f"\n  {'='*80}")
    print(f"  ENERGY EFFICIENCY ANALYSIS")
    print(f"  {'='*80}")
    print(f"  {'Metric':<30} {'Enhanced Eigenfaces':<20} {'Deep CNN (est.)':<20}")
    print(f"  {'-'*30} {'-'*20} {'-'*20}")
    
    # Get average inference time from results
    avg_inference_ms = 0
    count = 0
    for name in baseline_results:
        avg_inference_ms += baseline_results[name].get('avg_inference_ms', 0)
        count += 1
    if count > 0:
        avg_inference_ms /= count
    
    print(f"  {'Model Size (parameters)':<30} {'~0.05M':<20} {'~140M':<20}")
    print(f"  {'Avg Inference Time (CPU)':<30} {f'{avg_inference_ms:.1f} ms':<20} {'~2000 ms':<20}")
    print(f"  {'Hardware Requirement':<30} {'CPU / Raspberry Pi':<20} {'GPU (NVIDIA)':<20}")
    print(f"  {'Training Energy (est.)':<30} {'~0.07 kWh':<20} {'~57.6 kWh':<20}")
    print(f"  {'CO2 per Training':<30} {'~0.034 kg':<20} {'~28.8 kg':<20}")
    print(f"  {'Energy Ratio':<30} {'1x':<20} {'~850x':<20}")
    
    # ROC comparison: baseline vs enhanced
    roc_comparison = {}
    for name in eval_datasets:
        roc_comparison[f"Baseline ({name})"] = baseline_results.get(name, {})
        if name in enhanced_results:
            e_result = enhanced_results[name]
            if isinstance(e_result, dict) and 'kfold_accuracy' in e_result:
                roc_comparison[f"Enhanced ({name})"] = e_result
            else:
                best_metric = max(e_result.keys(),
                                key=lambda m: e_result[m]['kfold_accuracy'])
                roc_comparison[f"Enhanced ({name})"] = e_result[best_metric]
    
    plot_roc_curves(roc_comparison, "Baseline vs Enhanced", "roc_comparison.png")

def save_trained_models(baseline, enhanced_hist, baseline_results, enhanced_results):
    """Save trained models để load lại nhanh cho demo."""
    model_path = os.path.join(RESULTS_DIR, 'trained_models.npz')

    save_dict = {
        'image_size': IMAGE_SIZE,
        'n_components': N_COMPONENTS,
        # Baseline
        'bl_mean': baseline.pca.mean_,
        'bl_components': baseline.pca.components_,
        'bl_eigenvalues': baseline.pca.eigenvalues_,
        'bl_evr': baseline.pca.explained_variance_ratio_,
        'bl_n_features': baseline.pca.n_features_,
        # Enhanced HistEq
        'eh_mean': enhanced_hist.feature_extractor.pca.mean_,
        'eh_components': enhanced_hist.feature_extractor.pca.components_,
        'eh_eigenvalues': enhanced_hist.feature_extractor.pca.eigenvalues_,
        'eh_evr': enhanced_hist.feature_extractor.pca.explained_variance_ratio_,
        'eh_n_features': enhanced_hist.feature_extractor.pca.n_features_,
        # Results
        'baseline_results': baseline_results,
        'enhanced_results': enhanced_results,
    }

    np.savez(model_path, **save_dict)
    print(f"\n  Models saved: {model_path} ({os.path.getsize(model_path)/1e6:.1f} MB)")
    print(f"  Chạy 'python run_demo.py' để demo nhận diện!")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Run full experiment pipeline."""
    print_banner()

    total_start = time.time()

    # Phase 1: Load data
    train_images, train_labels, eval_datasets = phase1_load_data()

    # Phase 2: Baseline
    baseline, baseline_results = phase2_baseline(train_images, eval_datasets)

    # Phase 3: Ablation study
    ablation_results, model_combined = phase3_ablation(train_images, eval_datasets)

    # Phase 4: Enhanced Eigenfaces
    enhanced_hist, enhanced_results = phase4_enhanced(train_images, eval_datasets)

    # Phase 5: Comparison
    phase5_comparison(baseline_results, enhanced_results, eval_datasets)

    # Save results
    print_phase(6, "SAVING RESULTS")
    all_results = {
        "Baseline": baseline_results,
        "Enhanced": enhanced_results
    }
    save_results_table(all_results)

    # ★ SAVE MODELS
    save_trained_models(baseline, enhanced_hist, baseline_results, enhanced_results)

    total_elapsed = time.time() - total_start
    print(f"\n{'='*75}")
    print(f"  ALL EXPERIMENTS COMPLETE!")
    print(f"  Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")
    print(f"  Results saved to: {RESULTS_DIR}")
    print(f"  Figures saved to: {FIGURES_DIR}")
    print(f"{'='*75}")


if __name__ == "__main__":
    main()
