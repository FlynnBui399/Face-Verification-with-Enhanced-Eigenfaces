"""
Distance Metrics - Các hàm khoảng cách cho face verification (FROM SCRATCH).

╔══════════════════════════════════════════════════════════════════════════╗
║  TẤT CẢ DISTANCE METRICS IMPLEMENT TỪ ĐẦU                             ║
║  KHÔNG dùng scipy.spatial.distance                                      ║
║  KHÔNG dùng sklearn.metrics.pairwise                                    ║
╚══════════════════════════════════════════════════════════════════════════╝

Các metrics:
  1. Euclidean Distance: d(a,b) = ‖a - b‖₂
  2. Cosine Distance: d(a,b) = 1 - (a·b)/(‖a‖·‖b‖)
  3. Manhattan Distance: d(a,b) = Σ|aᵢ - bᵢ|
  4. Chi-Square Distance: d(a,b) = Σ(aᵢ - bᵢ)²/(aᵢ + bᵢ)
  5. Ensemble Metric Fusion: d = Σ wₖ·dₖ (weighted combination)

Author: Mathematics for AI - Final Project
"""

import numpy as np
from itertools import product as iter_product


# ==============================================================================
# 1. EUCLIDEAN DISTANCE (L2 norm)
# ==============================================================================

def euclidean_distance(a, b):
    """
    Euclidean Distance (L2 Distance).
    
    Công thức: d(a, b) = √(Σᵢ (aᵢ - bᵢ)²) = ‖a - b‖₂
    
    Ý nghĩa hình học: Khoảng cách đường thẳng trong không gian Euclid.
    Trong PCA space: phản ánh sự khác biệt tổng thể giữa 2 face embeddings.
    
    Properties:
        - d ≥ 0 (non-negativity)
        - d(a,a) = 0 (identity)
        - d(a,b) = d(b,a) (symmetry)
        - d(a,c) ≤ d(a,b) + d(b,c) (triangle inequality)
    
    Args:
        a, b: np.ndarray shape (d,) - feature vectors
    
    Returns:
        float: Khoảng cách Euclidean
    """
    diff = a - b
    return np.sqrt(np.sum(diff * diff))


def euclidean_distance_batch(A, B):
    """
    Batch Euclidean distance giữa pairs (A[i], B[i]).
    
    Args:
        A, B: np.ndarray shape (n, d)
    
    Returns:
        np.ndarray shape (n,)
    """
    diff = A - B
    return np.sqrt(np.sum(diff * diff, axis=1))


# ==============================================================================
# 2. COSINE DISTANCE
# ==============================================================================

def cosine_distance(a, b):
    """
    Cosine Distance.
    
    Công thức: d(a, b) = 1 - cos(θ) = 1 - (a · b) / (‖a‖ · ‖b‖)
    
    Trong đó cos(θ) = cosine similarity ∈ [-1, 1]
    → Cosine distance ∈ [0, 2]
    
    Ý nghĩa:
        - Đo góc giữa 2 vectors, KHÔNG phụ thuộc magnitude
        - d = 0: cùng hướng (identical faces)
        - d = 1: vuông góc (unrelated faces)
        - d = 2: ngược hướng
    
    Ưu điểm cho face recognition:
        - Invariant to scaling of feature vectors
        - Robust khi PCA projections có magnitude khác nhau
    
    Args:
        a, b: np.ndarray shape (d,)
    
    Returns:
        float: Cosine distance ∈ [0, 2]
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 1.0  # Undefined → treat as maximum distance
    
    cosine_sim = dot_product / (norm_a * norm_b)
    # Clamp to [-1, 1] for numerical stability
    cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
    
    return 1.0 - cosine_sim


def cosine_distance_batch(A, B):
    """Batch cosine distance."""
    dot = np.sum(A * B, axis=1)
    norm_a = np.linalg.norm(A, axis=1)
    norm_b = np.linalg.norm(B, axis=1)
    
    denom = norm_a * norm_b
    denom = np.maximum(denom, 1e-12)
    
    sim = np.clip(dot / denom, -1.0, 1.0)
    return 1.0 - sim


# ==============================================================================
# 3. MANHATTAN DISTANCE (L1 norm)
# ==============================================================================

def manhattan_distance(a, b):
    """
    Manhattan Distance (L1 Distance, City Block Distance).
    
    Công thức: d(a, b) = Σᵢ |aᵢ - bᵢ| = ‖a - b‖₁
    
    Ý nghĩa: Khoảng cách "đi theo phố" (chỉ đi ngang/dọc).
    
    So sánh với Euclidean:
        - Manhattan ít nhạy cảm với outliers (individual large differences)
        - Euclidean penalize large differences nhiều hơn (do squaring)
    
    Trong face recognition:
        - Robust hơn khi có một số PCA components bị noise
    
    Args:
        a, b: np.ndarray shape (d,)
    
    Returns:
        float: Manhattan distance
    """
    return np.sum(np.abs(a - b))


def manhattan_distance_batch(A, B):
    """Batch Manhattan distance."""
    return np.sum(np.abs(A - B), axis=1)


# ==============================================================================
# 4. CHI-SQUARE DISTANCE
# ==============================================================================

def chi_square_distance(a, b):
    """
    Chi-Square Distance.
    
    Công thức: d(a, b) = Σᵢ (aᵢ - bᵢ)² / (aᵢ + bᵢ)
    
    Ý nghĩa:
        - Đo sự khác biệt tương đối (relative difference)
        - Weight inversely proportional to (aᵢ + bᵢ)
        - Vùng có giá trị nhỏ: khác biệt nhỏ vẫn đáng kể
        - Vùng có giá trị lớn: khác biệt lớn mới đáng kể
    
    Nguồn gốc: Từ Chi-squared test trong thống kê
    Thường dùng cho histogram comparison → phù hợp PCA features
    
    Lưu ý: Cần shift features sang non-negative trước khi dùng.
    
    Args:
        a, b: np.ndarray shape (d,)
    
    Returns:
        float: Chi-square distance
    """
    # Shift to non-negative (add minimum)
    a_shifted = a - min(np.min(a), np.min(b)) + 1e-10
    b_shifted = b - min(np.min(a), np.min(b)) + 1e-10
    
    denominator = a_shifted + b_shifted
    # Avoid division by zero
    mask = denominator > 1e-12
    
    numerator = (a_shifted - b_shifted) ** 2
    
    distance = np.sum(numerator[mask] / denominator[mask])
    
    return distance


def chi_square_distance_batch(A, B):
    """Batch Chi-square distance."""
    # Shift per pair
    min_vals = np.minimum(np.min(A, axis=1, keepdims=True), 
                           np.min(B, axis=1, keepdims=True))
    A_s = A - min_vals + 1e-10
    B_s = B - min_vals + 1e-10
    
    denom = A_s + B_s
    denom = np.maximum(denom, 1e-12)
    
    numer = (A_s - B_s) ** 2
    return np.sum(numer / denom, axis=1)


# ==============================================================================
# 5. METRIC REGISTRY
# ==============================================================================

METRIC_FUNCTIONS = {
    "euclidean": euclidean_distance_batch,
    "cosine": cosine_distance_batch,
    "manhattan": manhattan_distance_batch,
    "chi_square": chi_square_distance_batch,
}


def compute_distances(features1, features2, metric_name="euclidean"):
    """
    Compute pairwise distances giữa 2 sets of features.
    
    Args:
        features1: np.ndarray shape (n, d)
        features2: np.ndarray shape (n, d)
        metric_name: "euclidean", "cosine", "manhattan", "chi_square"
    
    Returns:
        np.ndarray shape (n,) - distances
    """
    func = METRIC_FUNCTIONS[metric_name]
    return func(features1, features2)


# ==============================================================================
# 6. ENSEMBLE METRIC FUSION
# ==============================================================================

class EnsembleMetric:
    """
    Ensemble Distance Metric Fusion.
    
    Kết hợp nhiều distance metrics với learned weights:
        d_ensemble = Σₖ wₖ * normalize(dₖ)
    
    Normalization:
        - Min-max normalize mỗi metric về [0, 1] trước khi combine
        - Tránh metric có range lớn dominate
    
    Weight Learning:
        - Grid search trên validation set
        - Tìm weights {w₁, w₂, w₃, w₄} tối ưu verification accuracy
    """
    
    def __init__(self, metric_names=None, weights=None):
        self.metric_names = metric_names or ["euclidean", "cosine", "manhattan", "chi_square"]
        self.weights = weights or [0.25] * len(self.metric_names)
        self.weights = np.array(self.weights, dtype=np.float64)
        
        # Normalization parameters (learned during fit)
        self.min_vals = {}
        self.max_vals = {}
    
    def compute_all_distances(self, features1, features2):
        """
        Compute all metric distances.
        
        Returns:
            dict: {metric_name: distances array}
        """
        distances = {}
        for name in self.metric_names:
            distances[name] = compute_distances(features1, features2, name)
        return distances
    
    def normalize_distances(self, distances_dict, fit=False):
        """
        Min-max normalize distances về [0, 1].
        
        Args:
            distances_dict: {metric_name: distances}
            fit: True = learn min/max, False = use learned values
        """
        normalized = {}
        for name, dists in distances_dict.items():
            if fit:
                self.min_vals[name] = np.min(dists)
                self.max_vals[name] = np.max(dists)
            
            min_v = self.min_vals.get(name, np.min(dists))
            max_v = self.max_vals.get(name, np.max(dists))
            
            range_v = max_v - min_v
            if range_v > 1e-12:
                normalized[name] = (dists - min_v) / range_v
            else:
                normalized[name] = np.zeros_like(dists)
        
        return normalized
    
    def combine_distances(self, normalized_dict):
        """
        Weighted combination of normalized distances.
        
        d_ensemble = Σₖ wₖ * dₖ_normalized
        """
        combined = np.zeros(len(next(iter(normalized_dict.values()))), dtype=np.float64)
        
        for i, name in enumerate(self.metric_names):
            combined += self.weights[i] * normalized_dict[name]
        
        return combined
    
    def __call__(self, features1, features2, fit=False):
        """Compute ensemble distance."""
        distances = self.compute_all_distances(features1, features2)
        normalized = self.normalize_distances(distances, fit=fit)
        return self.combine_distances(normalized)
    
    def learn_weights(self, features1, features2, issame, n_folds=10):
        """
        Learn optimal weights using accuracy-weighted combination.
        
        Strategy (robust, avoids overfitting):
            1. Compute each metric's distances
            2. Find best threshold and accuracy for each metric individually
            3. Set weight proportional to accuracy (better metric = higher weight)
            4. Normalize weights to sum to 1.0
        
        This is more robust than exhaustive grid search because:
            - Each metric is evaluated independently
            - No combinatorial optimization that can overfit
            - Weights naturally reflect metric quality
        
        Args:
            features1, features2: np.ndarray shape (n, d) - pair features
            issame: np.ndarray of booleans - pair labels
            n_folds: Number of folds for threshold finding
        """
        print("\n  Learning ensemble weights (accuracy-weighted)...")
        
        # Compute all distances
        distances = self.compute_all_distances(features1, features2)
        normalized = self.normalize_distances(distances, fit=True)
        
        # Evaluate each metric individually
        metric_accuracies = {}
        for name in self.metric_names:
            dists = normalized[name]
            best_acc = 0.0
            for thresh in np.linspace(np.min(dists), np.max(dists), 200):
                predictions = dists < thresh
                acc = np.mean(predictions == issame)
                best_acc = max(best_acc, acc)
            metric_accuracies[name] = best_acc
            print(f"    {name}: {best_acc*100:.2f}%")
        
        # Set weights proportional to accuracy
        # Use (acc - 0.5) to weight relative to random baseline
        raw_weights = np.array([max(metric_accuracies[name] - 0.5, 0.01) 
                                for name in self.metric_names])
        self.weights = raw_weights / np.sum(raw_weights)
        
        # Also try: just use top-2 metrics
        sorted_metrics = sorted(metric_accuracies.items(), key=lambda x: -x[1])
        top2_names = [m[0] for m in sorted_metrics[:2]]
        
        print(f"  Learned weights: {dict(zip(self.metric_names, self.weights))}")
        print(f"  Top-2 metrics: {top2_names}")
        
        # Compute combined accuracy with learned weights
        combined = self.combine_distances(normalized)
        best_combined_acc = 0.0
        for thresh in np.linspace(np.min(combined), np.max(combined), 200):
            predictions = combined < thresh
            acc = np.mean(predictions == issame)
            best_combined_acc = max(best_combined_acc, acc)
        
        print(f"  Ensemble accuracy: {best_combined_acc*100:.2f}%")
        
        return best_combined_acc
