"""
Evaluation Protocols - Đánh giá face verification performance.

Protocols:
  1. Pair-wise Verification (LFW protocol)
  2. K-fold Cross-Validation
  3. ROC Curve & AUC Computation
  4. Threshold Optimization

Metrics:
  - Accuracy (ACC)
  - True Positive Rate (TPR) / False Positive Rate (FPR)
  - Area Under ROC Curve (AUC)
  - Equal Error Rate (EER)

All metrics computed FROM SCRATCH (không dùng sklearn.metrics).

Author: Mathematics for AI - Final Project
"""

import numpy as np
import time


# ==============================================================================
# 1. THRESHOLD-BASED VERIFICATION
# ==============================================================================

def verify_at_threshold(distances, issame, threshold):
    """
    Face verification tại một threshold cụ thể.
    
    Rule:
        - distance < threshold → predict "same person" (positive)
        - distance >= threshold → predict "different person" (negative)
    
    Args:
        distances: np.ndarray shape (N,) - distances giữa các pairs
        issame: np.ndarray shape (N,) - ground truth (True = same person)
        threshold: float - decision threshold
    
    Returns:
        dict: {accuracy, tpr, fpr, tp, fp, tn, fn}
    """
    predictions = distances < threshold  # True = same person
    
    # Confusion matrix
    tp = np.sum(predictions & issame)          # True positive
    fp = np.sum(predictions & ~issame)         # False positive
    tn = np.sum(~predictions & ~issame)        # True negative
    fn = np.sum(~predictions & issame)         # False negative
    
    total = len(issame)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    
    # TPR = TP / (TP + FN) = sensitivity = recall
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # FPR = FP / (FP + TN) = 1 - specificity
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'tpr': tpr,
        'fpr': fpr,
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn)
    }


def find_best_threshold(distances, issame, n_thresholds=200):
    """
    Tìm threshold tối ưu (maximize accuracy).
    
    Args:
        distances: np.ndarray shape (N,)
        issame: np.ndarray shape (N,)
        n_thresholds: Số thresholds thử
    
    Returns:
        tuple: (best_threshold, best_accuracy)
    """
    thresholds = np.linspace(np.min(distances), np.max(distances), n_thresholds)
    
    best_acc = 0.0
    best_thresh = 0.0
    
    for thresh in thresholds:
        result = verify_at_threshold(distances, issame, thresh)
        if result['accuracy'] > best_acc:
            best_acc = result['accuracy']
            best_thresh = thresh
    
    return best_thresh, best_acc


# ==============================================================================
# 2. K-FOLD CROSS-VALIDATION (LFW PROTOCOL)
# ==============================================================================

def kfold_verification(distances, issame, n_folds=10, n_thresholds=200):
    """
    K-fold cross-validation cho face verification.
    
    LFW Standard Protocol:
        - 6000 pairs chia thành 10 folds (600 pairs mỗi fold)
        - For each fold:
            1. Train: tìm optimal threshold trên 9 folds còn lại
            2. Test: evaluate accuracy trên fold hiện tại
        - Report: mean accuracy ± std across 10 folds
    
    Args:
        distances: np.ndarray shape (N,) - pair distances
        issame: np.ndarray shape (N,) - pair labels
        n_folds: Number of folds (default 10)
        n_thresholds: Thresholds to try for optimization
    
    Returns:
        dict: {
            'accuracy': mean accuracy,
            'std': standard deviation,
            'fold_accuracies': list of per-fold accuracies,
            'thresholds': list of per-fold thresholds
        }
    """
    N = len(distances)
    fold_size = N // n_folds
    
    # Shuffle indices (with fixed seed for reproducibility)
    rng = np.random.RandomState(42)
    indices = rng.permutation(N)
    
    fold_accuracies = []
    fold_thresholds = []
    
    for fold in range(n_folds):
        # Test indices for this fold
        test_start = fold * fold_size
        test_end = test_start + fold_size
        test_idx = indices[test_start:test_end]
        
        # Train indices (all other folds)
        train_idx = np.concatenate([indices[:test_start], indices[test_end:]])
        
        # Find best threshold on training folds
        train_distances = distances[train_idx]
        train_labels = issame[train_idx]
        best_thresh, _ = find_best_threshold(train_distances, train_labels, n_thresholds)
        
        # Evaluate on test fold
        test_distances = distances[test_idx]
        test_labels = issame[test_idx]
        result = verify_at_threshold(test_distances, test_labels, best_thresh)
        
        fold_accuracies.append(result['accuracy'])
        fold_thresholds.append(best_thresh)
    
    return {
        'accuracy': np.mean(fold_accuracies),
        'std': np.std(fold_accuracies),
        'fold_accuracies': fold_accuracies,
        'thresholds': fold_thresholds
    }


# ==============================================================================
# 3. ROC CURVE (FROM SCRATCH)
# ==============================================================================

def compute_roc(distances, issame, n_thresholds=500):
    """
    Compute ROC (Receiver Operating Characteristic) curve.
    
    ROC curve: plot TPR vs FPR tại các thresholds khác nhau.
    
    Ý nghĩa:
        - Mỗi điểm (FPR, TPR) tương ứng với một threshold
        - Threshold thấp → classify nhiều positive → high TPR, high FPR
        - Threshold cao → classify ít positive → low TPR, low FPR
        - ROC tốt: gần góc trên trái (TPR=1, FPR=0)
    
    Args:
        distances: np.ndarray shape (N,)
        issame: np.ndarray shape (N,)
        n_thresholds: Số điểm trên ROC curve
    
    Returns:
        dict: {'fpr': array, 'tpr': array, 'thresholds': array, 'auc': float}
    """
    thresholds = np.linspace(np.min(distances) - 0.01, np.max(distances) + 0.01, n_thresholds)
    
    tpr_list = []
    fpr_list = []
    
    for thresh in thresholds:
        result = verify_at_threshold(distances, issame, thresh)
        tpr_list.append(result['tpr'])
        fpr_list.append(result['fpr'])
    
    tpr_arr = np.array(tpr_list)
    fpr_arr = np.array(fpr_list)
    
    # Compute AUC using trapezoidal rule (from scratch)
    # AUC = ∫ TPR d(FPR)
    # Sort by FPR for proper integration
    sorted_idx = np.argsort(fpr_arr)
    fpr_sorted = fpr_arr[sorted_idx]
    tpr_sorted = tpr_arr[sorted_idx]
    
    # Trapezoidal rule: Σ (x_{i+1} - x_i) * (y_i + y_{i+1}) / 2
    auc = 0.0
    for i in range(len(fpr_sorted) - 1):
        dx = fpr_sorted[i+1] - fpr_sorted[i]
        avg_y = (tpr_sorted[i] + tpr_sorted[i+1]) / 2.0
        auc += dx * avg_y
    
    return {
        'fpr': fpr_arr,
        'tpr': tpr_arr,
        'thresholds': thresholds,
        'auc': auc
    }


# ==============================================================================
# 4. EQUAL ERROR RATE (FROM SCRATCH)
# ==============================================================================

def compute_eer(distances, issame, n_thresholds=500):
    """
    Compute Equal Error Rate (EER).
    
    EER = threshold tại đó FPR = FNR (False Negative Rate = 1 - TPR)
    
    Ý nghĩa:
        - EER thấp → model tốt
        - EER = 0.0: perfect classifier
        - EER = 0.5: random classifier
    
    Returns:
        tuple: (eer, threshold_at_eer)
    """
    roc = compute_roc(distances, issame, n_thresholds)
    
    # Find where FPR ≈ FNR (i.e., FPR ≈ 1 - TPR)
    fnr = 1.0 - roc['tpr']
    min_diff = float('inf')
    eer = 0.0
    eer_thresh = 0.0
    
    for i in range(len(roc['fpr'])):
        diff = abs(roc['fpr'][i] - fnr[i])
        if diff < min_diff:
            min_diff = diff
            eer = (roc['fpr'][i] + fnr[i]) / 2.0
            eer_thresh = roc['thresholds'][i]
    
    return eer, eer_thresh


# ==============================================================================
# 5. FULL EVALUATION PIPELINE
# ==============================================================================

def evaluate_model(model, eval_images, issame, dataset_name="Dataset",
                   metric="euclidean", verbose=True):
    """
    Full evaluation pipeline cho một model trên một eval dataset.
    
    Args:
        model: Object có method extract_features(images)
        eval_images: np.ndarray shape (2N, H, W) - pairs
        issame: np.ndarray shape (N,) - pair labels
        dataset_name: Name for logging
        metric: Distance metric to use
        verbose: Print results
    
    Returns:
        dict: Full evaluation results
    """
    start_time = time.time()
    
    n_pairs = len(issame)
    
    # Split into pairs
    images1 = eval_images[0::2]  # Even indices
    images2 = eval_images[1::2]  # Odd indices
    
    # Truncate to match
    min_len = min(len(images1), len(images2), n_pairs)
    images1 = images1[:min_len]
    images2 = images2[:min_len]
    issame = issame[:min_len]
    
    if verbose:
        print(f"\n  Evaluating on {dataset_name}: {min_len} pairs")
    
    # Extract features
    feat_start = time.time()
    features1 = model.extract_features(images1)
    features2 = model.extract_features(images2)
    feat_time = time.time() - feat_start
    
    # Compute distances
    if metric == "ensemble" and hasattr(model, 'ensemble_metric'):
        distances = model.ensemble_metric(features1, features2)
    else:
        from src.metrics import compute_distances
        distances = compute_distances(features1, features2, metric)
    
    # K-fold verification
    kfold_result = kfold_verification(distances, issame, n_folds=10)
    
    # ROC curve
    roc_result = compute_roc(distances, issame)
    
    # EER
    eer, eer_thresh = compute_eer(distances, issame)
    
    # Best single threshold
    best_thresh, best_acc = find_best_threshold(distances, issame)
    
    total_time = time.time() - start_time
    avg_inference = feat_time / (2 * min_len)  # per image
    
    results = {
        'dataset': dataset_name,
        'metric': metric,
        'n_pairs': min_len,
        'kfold_accuracy': kfold_result['accuracy'],
        'kfold_std': kfold_result['std'],
        'best_accuracy': best_acc,
        'best_threshold': best_thresh,
        'auc': roc_result['auc'],
        'eer': eer,
        'fold_accuracies': kfold_result['fold_accuracies'],
        'roc': roc_result,
        'avg_inference_ms': avg_inference * 1000,
        'total_time_s': total_time,
        'distances': distances,
        'issame': issame,
    }
    
    if verbose:
        print(f"  {dataset_name} Results ({metric}):")
        print(f"    10-Fold Accuracy: {kfold_result['accuracy']*100:.2f}% "
              f"± {kfold_result['std']*100:.2f}%")
        print(f"    Best Accuracy:    {best_acc*100:.2f}% (thresh={best_thresh:.4f})")
        print(f"    AUC:              {roc_result['auc']:.4f}")
        print(f"    EER:              {eer*100:.2f}%")
        print(f"    Avg Inference:    {avg_inference*1000:.2f} ms/image")
    
    return results


def evaluate_all_metrics(model, eval_images, issame, dataset_name="Dataset", verbose=True):
    """
    Evaluate model với tất cả distance metrics.
    
    Returns:
        dict: {metric_name: results}
    """
    metrics_to_test = ["euclidean", "cosine", "manhattan", "chi_square"]
    
    all_results = {}
    
    for metric in metrics_to_test:
        results = evaluate_model(model, eval_images, issame, 
                                dataset_name, metric, verbose)
        all_results[metric] = results
    
    # Find best metric
    best_metric = max(all_results.keys(), 
                      key=lambda m: all_results[m]['kfold_accuracy'])
    
    if verbose:
        print(f"\n  Best metric for {dataset_name}: {best_metric} "
              f"({all_results[best_metric]['kfold_accuracy']*100:.2f}%)")
    
    return all_results


def print_comparison_table(results_dict, dataset_name="LFW"):
    """
    In bảng so sánh kết quả các phương pháp.
    
    Args:
        results_dict: {method_name: {metric: results_dict}}
    """
    print(f"\n{'='*80}")
    print(f"  COMPARISON TABLE - {dataset_name}")
    print(f"{'='*80}")
    print(f"  {'Method':<35} {'Metric':<12} {'Acc(%)':<10} {'±Std':<8} {'AUC':<8} {'EER(%)':<8}")
    print(f"  {'-'*35} {'-'*12} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
    
    for method_name, metric_results in results_dict.items():
        if isinstance(metric_results, dict) and 'kfold_accuracy' in metric_results:
            # Single metric result
            r = metric_results
            print(f"  {method_name:<35} {r.get('metric','euc'):<12} "
                  f"{r['kfold_accuracy']*100:<10.2f} "
                  f"{r['kfold_std']*100:<8.2f} "
                  f"{r['auc']:<8.4f} "
                  f"{r['eer']*100:<8.2f}")
        else:
            # Multi-metric results
            for metric_name, r in metric_results.items():
                label = f"{method_name} ({metric_name})"
                print(f"  {label:<35} {metric_name:<12} "
                      f"{r['kfold_accuracy']*100:<10.2f} "
                      f"{r['kfold_std']*100:<8.2f} "
                      f"{r['auc']:<8.4f} "
                      f"{r['eer']*100:<8.2f}")
    
    print(f"{'='*80}")
