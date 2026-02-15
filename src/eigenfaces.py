"""
Eigenfaces - Baseline và Enhanced Eigenfaces implementations.

╔══════════════════════════════════════════════════════════════════════════╗
║  Baseline: Standard Eigenfaces (Turk & Pentland, 1991)                  ║
║  Enhanced: Multi-scale + Ensemble Metrics + Illumination Normalization  ║
║  Tất cả code FROM SCRATCH, không dùng sklearn/dlib/face_recognition    ║
╚══════════════════════════════════════════════════════════════════════════╝

Classes:
  1. BaselineEigenfaces - Standard PCA-based face recognition
  2. MultiScaleEigenfaces - Multi-scale pyramid feature extraction
  3. EnhancedEigenfaces - Full Enhanced method (all improvements)

Reference:
  - Turk, M., & Pentland, A. (1991). "Eigenfaces for recognition"
  - Tan, X., & Triggs, B. (2010). "Enhanced local texture feature sets"

Author: Mathematics for AI - Final Project
"""

import numpy as np
from tqdm import tqdm

from src.linalg_scratch import PCA_Scratch
from src.preprocessing import (
    preprocess_batch, resize_batch, standardize,
    histogram_equalization, tan_triggs_normalization
)
from src.metrics import (
    euclidean_distance_batch, cosine_distance_batch,
    manhattan_distance_batch, chi_square_distance_batch,
    EnsembleMetric
)


# ==============================================================================
# 1. BASELINE EIGENFACES
# ==============================================================================

class BaselineEigenfaces:
    """
    Standard Eigenfaces (Turk & Pentland, 1991).
    
    Thuật toán:
        Training:
            1. Flatten images: (N, H, W) → (N, H*W)
            2. PCA from scratch: tìm top-K eigenvectors (eigenfaces)
            3. Project training images → K-dimensional space
        
        Verification (given pair of images):
            1. Preprocess & flatten
            2. Project → PCA space
            3. Compute distance
            4. Compare with threshold → same/different person
    
    Đây là BASELINE để so sánh với Enhanced Eigenfaces.
    """
    
    def __init__(self, n_components=200, image_size=64):
        self.n_components = n_components
        self.image_size = image_size
        self.pca = PCA_Scratch(n_components=n_components)
        self.name = "Baseline Eigenfaces"
    
    def fit(self, images, verbose=True):
        """
        Train eigenfaces model.
        
        Args:
            images: np.ndarray shape (N, H, W) - training images
            verbose: Show progress
        
        Returns:
            self
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"  Training: {self.name}")
            print(f"  Images: {images.shape[0]}, Size: {images.shape[1]}x{images.shape[2]}")
            print(f"  Components: {self.n_components}")
            print(f"{'='*60}")
        
        N = images.shape[0]
        d = self.image_size * self.image_size
        
        # Flatten: (N, H, W) → (N, H*W) - copy to avoid modifying original
        # Use input dtype (float32 or float64) — float32 saves 50% RAM
        X = images.reshape(N, d).copy()
        
        # Standardize each image
        for i in range(N):
            X[i] = standardize(X[i])
        
        # Free original images reference to save memory
        # PCA from scratch
        self.pca.fit(X, verbose=verbose)
        del X  # Free memory after fit
        
        return self
    
    def extract_features(self, images):
        """
        Extract PCA features from images.
        
        Args:
            images: np.ndarray shape (N, H, W) or (H, W)
        
        Returns:
            np.ndarray: shape (N, n_components) or (n_components,)
        """
        single = False
        if images.ndim == 2:
            images = images[np.newaxis, ...]
            single = True
        
        N = images.shape[0]
        d = self.image_size * self.image_size
        
        # Flatten and standardize
        X = images.reshape(N, d).copy()
        for i in range(N):
            X[i] = standardize(X[i])
        
        # Project to PCA space
        features = self.pca.transform(X)
        
        if single:
            return features[0]
        return features
    
    def get_eigenfaces(self):
        """
        Lấy eigenfaces (principal components) reshape về dạng ảnh.
        
        Returns:
            np.ndarray: shape (n_components, H, W)
        """
        return self.pca.components_.reshape(-1, self.image_size, self.image_size)
    
    def reconstruct(self, images, n_components=None):
        """
        Reconstruct images từ PCA features (dùng raw projection).
        
        Args:
            images: np.ndarray shape (N, H, W)
            n_components: Số components dùng (None = all)
        
        Returns:
            np.ndarray: shape (N, H, W) - reconstructed images
        """
        single = False
        if images.ndim == 2:
            images = images[np.newaxis, ...]
            single = True
        
        N = images.shape[0]
        d = self.image_size * self.image_size
        X = images.reshape(N, d).copy()
        for i in range(N):
            X[i] = standardize(X[i])
        
        # Use RAW transform for reconstruction (no whitening/L2)
        features = self.pca.transform_raw(X)
        
        if n_components is not None and n_components < features.shape[-1]:
            features[:, n_components:] = 0
        
        reconstructed = self.pca.inverse_transform(features)
        
        if single:
            return reconstructed[0].reshape(self.image_size, self.image_size)
        return reconstructed.reshape(-1, self.image_size, self.image_size)


# ==============================================================================
# 2. MULTI-SCALE EIGENFACES
# ==============================================================================

class MultiScaleEigenfaces:
    """
    Multi-Scale Pyramid Feature Extraction.
    
    Contribution C1: Extract eigenfaces từ nhiều scales.
    
    Thuật toán:
        1. Cho mỗi scale s ∈ {64, 32, 16}:
            a. Resize images → s × s
            b. Train PCA riêng cho scale này
            c. Extract features (n_components_s dimensions)
        
        2. Concatenate features từ tất cả scales:
            f_multi = [f_64 ; f_32 ; f_16]
            
    Ý nghĩa:
        - Scale lớn (64×64): Capture global structure (overall face shape)
        - Scale vừa (32×32): Capture mid-level features (eyes, nose, mouth regions)
        - Scale nhỏ (16×16): Capture coarse features (head shape, hair)
        
    Expected improvement: +3-5% accuracy, robust với pose variation
    """
    
    def __init__(self, scales=None, n_components_per_scale=None):
        """
        Args:
            scales: List of image sizes, e.g. [64, 32]
            n_components_per_scale: Components cho mỗi scale, 
                                     e.g. [200, 100]
        """
        self.scales = scales or [64, 32]
        self.n_components_per_scale = n_components_per_scale or [200, 100]
        
        # Validate
        assert len(self.scales) == len(self.n_components_per_scale)
        
        # PCA model cho mỗi scale
        self.pca_models = {}
        self.name = f"Multi-Scale Eigenfaces ({self.scales})"
    
    def fit(self, images, verbose=True):
        """
        Train PCA riêng cho mỗi scale.
        
        Args:
            images: np.ndarray shape (N, H, W) - original resolution images
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"  Training: {self.name}")
            print(f"  Scales: {self.scales}")
            print(f"  Components per scale: {self.n_components_per_scale}")
            print(f"{'='*60}")
        
        for scale, n_comp in zip(self.scales, self.n_components_per_scale):
            if verbose:
                print(f"\n  --- Scale {scale}×{scale}, {n_comp} components ---")
            
            # Resize to this scale
            if scale == images.shape[1]:
                scaled_images = images
            else:
                scaled_images = resize_batch(images, scale)
            
            # Flatten with copy — avoid modifying original images
            N = scaled_images.shape[0]
            d = scale * scale
            X = scaled_images.reshape(N, d).copy()
            
            # Standardize
            for i in range(N):
                X[i] = standardize(X[i])
            
            # Train PCA
            pca = PCA_Scratch(n_components=n_comp)
            pca.fit(X, verbose=verbose)
            
            self.pca_models[scale] = pca
            del X
            import gc; gc.collect()
        
        return self
    
    def extract_features(self, images):
        """
        Extract và concatenate multi-scale features.
        
        Args:
            images: np.ndarray shape (N, H, W) hoặc (H, W)
        
        Returns:
            np.ndarray: shape (N, total_components) - concatenated features
        """
        single = False
        if images.ndim == 2:
            images = images[np.newaxis, ...]
            single = True
        
        all_features = []
        
        for scale in self.scales:
            # Resize
            if scale == images.shape[1]:
                scaled = images
            else:
                scaled = resize_batch(images, scale)
            
            # Flatten, COPY, and standardize
            N = scaled.shape[0]
            d = scale * scale
            X = scaled.reshape(N, d).copy()
            for i in range(N):
                X[i] = standardize(X[i])
            
            # Project (with whitening + L2 norm)
            features = self.pca_models[scale].transform(X)
            all_features.append(features)
        
        # Concatenate: [f_64 ; f_32 ; f_16]
        concatenated = np.hstack(all_features)
        
        # L2 normalize the concatenated vector
        if concatenated.ndim == 1:
            norm = np.linalg.norm(concatenated)
            if norm > 1e-12:
                concatenated = concatenated / norm
        else:
            norms = np.linalg.norm(concatenated, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            concatenated = concatenated / norms
        
        if single:
            return concatenated[0]
        return concatenated


# ==============================================================================
# 3. ENHANCED EIGENFACES (COMBINED METHOD)
# ==============================================================================

class EnhancedEigenfaces:
    """
    Enhanced Eigenfaces - Full method với tất cả improvements.
    
    Contributions:
        C1. Multi-Scale Pyramid Feature Extraction
        C2. Ensemble Distance Metric Fusion  
        C3. Illumination Normalization Pipeline (Histogram Eq / Tan-Triggs)
    
    Pipeline:
        Training:
            1. Illumination normalization (C3: hist_eq or tan_triggs)
               → Applied to 2D images BEFORE flattening
               → Does NOT include standardize (handled by feature_extractor)
            2. Multi-scale PCA training (C1)
            3. Learn ensemble metric weights (C2, on validation set)
        
        Verification:
            1. Preprocess pair images (illumination norm)
            2. Extract multi-scale features
            3. Compute distance (cosine or ensemble)
            4. Threshold comparison
    
    IMPORTANT FIX: Previous version had a double-standardize bug where
    preprocess_batch() called standardize() AND feature_extractor also
    called standardize(). Now illumination normalization is applied
    directly (without standardize), letting the feature_extractor handle
    standardization exactly once.
    """
    
    def __init__(self, scales=None, n_components_per_scale=None,
                 use_illumination_norm=True, use_ensemble=False,
                 use_multiscale=True, illumination_method="hist_eq"):
        """
        Args:
            scales: Multi-scale sizes (default [64, 32])
            n_components_per_scale: Components per scale (default [200, 100])
            use_illumination_norm: Enable C3 (illumination normalization)
            use_ensemble: Enable C2 (ensemble metrics)
            use_multiscale: Enable C1 (multi-scale)
            illumination_method: "hist_eq" or "tan_triggs" (default "hist_eq")
        """
        self.use_illumination_norm = use_illumination_norm
        self.use_ensemble = use_ensemble
        self.use_multiscale = use_multiscale
        self.illumination_method = illumination_method if use_illumination_norm else None
        
        self.scales = scales or [64, 32]
        self.n_components_per_scale = n_components_per_scale or [200, 100]
        
        if use_multiscale:
            self.feature_extractor = MultiScaleEigenfaces(
                self.scales, self.n_components_per_scale
            )
        else:
            self.feature_extractor = BaselineEigenfaces(
                n_components=self.n_components_per_scale[0],
                image_size=self.scales[0]
            )
        
        if use_ensemble:
            self.ensemble_metric = EnsembleMetric()
        
        # Build name
        parts = ["Enhanced Eigenfaces"]
        if use_illumination_norm:
            method_label = "HistEq" if illumination_method == "hist_eq" else "TanTriggs"
            parts.append(f"+{method_label}")
        if use_multiscale:
            parts.append("+MultiScale")
        if use_ensemble:
            parts.append("+Ensemble")
        self.name = " ".join(parts)
    
    def _apply_illumination(self, images):
        """
        Apply illumination normalization to 2D images.
        
        IMPORTANT: This only applies the illumination normalization transform.
        It does NOT call standardize() — that is handled by the feature_extractor
        to avoid double-standardization.
        
        Args:
            images: np.ndarray shape (N, H, W)
        
        Returns:
            np.ndarray shape (N, H, W) - preprocessed images
        """
        if self.illumination_method is None:
            return images
        
        N = images.shape[0]
        processed = np.zeros_like(images)
        
        for i in range(N):
            if self.illumination_method == "tan_triggs":
                processed[i] = tan_triggs_normalization(images[i])
            else:  # "hist_eq"
                processed[i] = histogram_equalization(images[i])
        
        return processed
    
    def fit(self, images, verbose=True):
        """
        Train Enhanced Eigenfaces.
        
        Args:
            images: np.ndarray shape (N, H, W)
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"  Training: {self.name}")
            print(f"{'='*60}")
        
        # Step 1: Illumination normalization (applied to 2D images)
        # NO standardize here — feature_extractor handles it
        if self.use_illumination_norm:
            method_label = "Histogram Equalization" if self.illumination_method == "hist_eq" else "Tan-Triggs"
            if verbose:
                print(f"\n  [C3] Applying {method_label} illumination normalization...")
            processed = self._apply_illumination(images)
        else:
            processed = images
        
        # Step 2: Train feature extractor (handles standardize internally)
        self.feature_extractor.fit(processed, verbose=verbose)
        
        return self
    
    def extract_features(self, images):
        """
        Extract features with preprocessing.
        
        Pipeline:
            1. Apply illumination normalization (2D, no standardize)
            2. Feature extractor handles: flatten → standardize → PCA → whiten → L2 norm
        """
        single = False
        if images.ndim == 2:
            images = images[np.newaxis, ...]
            single = True
        
        # Illumination normalization (no standardize)
        processed = self._apply_illumination(images)
        
        features = self.feature_extractor.extract_features(processed)
        if single:
            return features[0] if features.ndim > 1 else features
        return features
    
    def learn_ensemble_weights(self, images1, images2, issame):
        """
        Learn ensemble metric weights on validation pairs.
        
        Args:
            images1, images2: np.ndarray shape (N, H, W) - pair images
            issame: np.ndarray of booleans
        """
        if not self.use_ensemble:
            return
        
        features1 = self.extract_features(images1)
        features2 = self.extract_features(images2)
        
        self.ensemble_metric.learn_weights(features1, features2, issame)
    
    def compute_distances(self, features1, features2, metric="euclidean"):
        """
        Compute distances between feature pairs.
        
        Args:
            features1, features2: np.ndarray shape (N, d)
            metric: "euclidean", "cosine", "manhattan", "chi_square", "ensemble"
        
        Returns:
            np.ndarray shape (N,)
        """
        if metric == "ensemble" and self.use_ensemble:
            return self.ensemble_metric(features1, features2)
        
        from src.metrics import compute_distances as _compute
        return _compute(features1, features2, metric)
