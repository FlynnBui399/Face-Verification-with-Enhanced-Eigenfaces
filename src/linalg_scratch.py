"""
Linear Algebra From Scratch - SVD và PCA implementation.

╔══════════════════════════════════════════════════════════════════════════╗
║  TẤT CẢ THUẬT TOÁN TRONG FILE NÀY ĐƯỢC IMPLEMENT TỪ ĐẦU              ║
║  KHÔNG sử dụng sklearn.decomposition.PCA                               ║
║  KHÔNG sử dụng numpy.linalg.svd cho PCA chính                          ║
║  Chỉ dùng numpy cho phép nhân ma trận cơ bản (BLAS operations)         ║
╚══════════════════════════════════════════════════════════════════════════╝

Các thuật toán:
  1. Gram-Schmidt Orthogonalization
  2. QR Decomposition (via Modified Gram-Schmidt)
  3. Power Iteration (tìm eigenvector chính)
  4. Eigendecomposition Top-K (Power Iteration + Deflation)
  5. Truncated SVD From Scratch
  6. PCA Class (from scratch)

Tham khảo:
  - Turk & Pentland (1991), "Eigenfaces for Recognition"
  - Golub & Van Loan (2013), "Matrix Computations" (4th ed.)
  - Halko et al. (2011), "Finding Structure with Randomness"

Author: Mathematics for AI - Final Project
"""

import numpy as np
from tqdm import tqdm
from src.config import USE_GPU

# ==============================================================================
# GPU ACCELERATION (CuPy - optional)
# ==============================================================================
GPU_AVAILABLE = False
cp = None

if USE_GPU:
    try:
        # On Windows, add NVIDIA DLL directories before importing CuPy
        import os, sys
        if sys.platform == 'win32':
            _site_pkgs = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            _site_pkgs = os.path.join(os.path.dirname(sys.executable), 
                                       'Lib', 'site-packages')
            _nvidia_dirs = [
                os.path.join(_site_pkgs, 'nvidia', 'cublas', 'bin'),
                os.path.join(_site_pkgs, 'nvidia', 'cuda_runtime', 'bin'),
                os.path.join(_site_pkgs, 'nvidia', 'cuda_nvrtc', 'bin'),
                os.path.join(_site_pkgs, 'nvidia', 'cusolver', 'bin'),
                os.path.join(_site_pkgs, 'nvidia', 'cusparse', 'bin'),
                os.path.join(_site_pkgs, 'nvidia', 'nvjitlink', 'bin'),
                os.path.join(_site_pkgs, 'nvidia', 'cufft', 'bin'),
                os.path.join(_site_pkgs, 'nvidia', 'curand', 'bin'),
            ]
            for _d in _nvidia_dirs:
                if os.path.isdir(_d):
                    os.add_dll_directory(_d)
        
        import cupy as _cp
        # Test that cuBLAS actually works (not just import)
        _test = _cp.asarray(np.array([1.0, 2.0]))
        _ = _cp.dot(_test, _test)
        del _test
        
        if _cp.cuda.is_available():
            cp = _cp
            GPU_AVAILABLE = True
            _free, _total = cp.cuda.Device(0).mem_info
            print(f"  [GPU] CuPy {cp.__version__} - CUDA GPU detected! "
                  f"({_free/1e9:.1f}/{_total/1e9:.1f} GB free)", flush=True)
        else:
            print("  [GPU] CUDA not available, using CPU", flush=True)
    except (ImportError, Exception) as e:
        print(f"  [GPU] GPU init failed ({e}), using CPU", flush=True)


# ==============================================================================
# 1. GRAM-SCHMIDT ORTHOGONALIZATION (FROM SCRATCH)
# ==============================================================================

def modified_gram_schmidt(V):
    """
    Modified Gram-Schmidt Orthogonalization.
    
    Biến tập vectors V = {v₁, v₂, ..., vₖ} thành tập orthonormal U = {u₁, u₂, ..., uₖ}
    
    Thuật toán Modified Gram-Schmidt (numerically stable hơn Classical):
        for i = 1 to k:
            uᵢ = vᵢ
            for j = 1 to i-1:
                uᵢ = uᵢ - <uᵢ, uⱼ> * uⱼ     # Trừ projection lên uⱼ
            uᵢ = uᵢ / ‖uᵢ‖                     # Normalize
    
    Sự khác biệt với Classical Gram-Schmidt:
        - Classical: proj lên u_j ban đầu → unstable khi vectors gần parallel
        - Modified: proj lên u_j đã update → numerically stable hơn
    
    Complexity: O(k²n) với k vectors, mỗi vector có n chiều
    
    Args:
        V: np.ndarray shape (k, n) - k vectors, mỗi vector n chiều
    
    Returns:
        U: np.ndarray shape (k, n) - k orthonormal vectors
    """
    k, n = V.shape
    U = V.copy().astype(np.float64)
    
    for i in range(k):
        # Trừ projection lên tất cả vectors trước đó
        for j in range(i):
            # <uᵢ, uⱼ> = dot product
            proj = np.dot(U[i], U[j])
            U[i] = U[i] - proj * U[j]
        
        # Normalize
        norm = np.linalg.norm(U[i])
        if norm > 1e-12:
            U[i] = U[i] / norm
        else:
            # Vector gần zero → đã linearly dependent
            U[i] = np.zeros(n)
    
    return U


def orthogonalize_against(v, U):
    """
    Orthogonalize vector v against set of orthonormal vectors U.
    
    v_orth = v - Σⱼ <v, uⱼ> * uⱼ
    
    Args:
        v: np.ndarray shape (n,) - vector cần orthogonalize
        U: np.ndarray shape (k, n) - k orthonormal vectors
    
    Returns:
        np.ndarray: v sau khi orthogonalize, đã normalize
    """
    v_orth = v.copy()
    for j in range(len(U)):
        proj = np.dot(v_orth, U[j])
        v_orth = v_orth - proj * U[j]
    
    norm = np.linalg.norm(v_orth)
    if norm > 1e-12:
        v_orth = v_orth / norm
    
    return v_orth


# ==============================================================================
# 2. POWER ITERATION (FROM SCRATCH)
# ==============================================================================

def power_iteration(A, max_iter=200, tol=1e-10, v0=None):
    """
    Power Iteration - Tìm eigenvector ứng với eigenvalue lớn nhất.
    
    Thuật toán:
        Input: Ma trận vuông A (n×n), symmetric positive semi-definite
        
        1. v₀ = random unit vector
        2. Repeat:
            a. w = A @ v        (nhân ma trận-vector)
            b. λ = vᵀ @ w       (Rayleigh quotient = eigenvalue estimate)
            c. v_new = w / ‖w‖  (normalize)
            d. If ‖v_new - v‖ < tol: break (converged)
            e. v = v_new
        3. Return (λ, v)
    
    Tại sao hoạt động:
        - Giả sử A có eigenvalues |λ₁| > |λ₂| ≥ ... ≥ |λₙ|
        - v₀ = c₁e₁ + c₂e₂ + ... + cₙeₙ (phân tích theo eigenvectors)
        - A^k v₀ = c₁λ₁ᵏe₁ + c₂λ₂ᵏe₂ + ... + cₙλₙᵏeₙ
        - Khi k → ∞: λ₁ᵏ dominate → A^k v₀ ≈ c₁λ₁ᵏe₁
        - Sau normalize: v → e₁ (eigenvector chính)
    
    Tốc độ hội tụ: |λ₂/λ₁|ᵏ → nhanh nếu eigenvalue gap lớn
    
    Complexity: O(n² × max_iter) - mỗi iteration là phép nhân matrix-vector O(n²)
    
    Args:
        A: np.ndarray shape (n, n) - symmetric matrix
        max_iter: Số iteration tối đa
        tol: Ngưỡng hội tụ
        v0: Vector khởi tạo (None = random)
    
    Returns:
        tuple: (eigenvalue, eigenvector)
    """
    n = A.shape[0]
    
    # Khởi tạo random vector
    if v0 is None:
        rng = np.random.RandomState(42)
        v = rng.randn(n)
    else:
        v = v0.copy()
    v = v / np.linalg.norm(v)
    
    eigenvalue = 0.0
    
    for iteration in range(max_iter):
        # w = A @ v (matrix-vector multiplication)
        w = A @ v
        
        # Rayleigh quotient: λ = vᵀAv = vᵀw
        eigenvalue = np.dot(v, w)
        
        # Normalize
        norm_w = np.linalg.norm(w)
        if norm_w < 1e-15:
            break
        v_new = w / norm_w
        
        # Check convergence: ‖v_new - v‖ < tol
        # Dùng abs vì eigenvector có thể flip sign
        diff = min(np.linalg.norm(v_new - v), np.linalg.norm(v_new + v))
        
        v = v_new
        
        if diff < tol:
            break
    
    return eigenvalue, v


# ==============================================================================
# 3. EIGENDECOMPOSITION TOP-K (POWER ITERATION + DEFLATION)
# ==============================================================================

def eigendecompose_top_k(A, k, max_iter=200, tol=1e-10, verbose=True):
    """
    Tính top-K eigenvalues và eigenvectors bằng Power Iteration + Deflation.
    
    Thuật toán:
        for i = 1 to K:
            1. (λᵢ, vᵢ) = power_iteration(A)          # Tìm eigen lớn nhất
            2. A = A - λᵢ * vᵢ @ vᵢᵀ                   # Deflation: loại bỏ component này
            3. Re-orthogonalize vᵢ against {v₁,...,vᵢ₋₁} # Đảm bảo orthogonal
    
    Deflation Matrix:
        - Sau khi tìm (λ₁, v₁), ta trừ đi component: A' = A - λ₁v₁v₁ᵀ
        - A' có eigenvalues {0, λ₂, λ₃, ..., λₙ} (λ₁ bị thay bằng 0)
        - Power iteration trên A' sẽ tìm λ₂
        - Lặp lại cho K components
    
    Re-orthogonalization:
        - Do numerical errors, eigenvectors có thể drift khỏi orthogonal
        - Modified Gram-Schmidt sau mỗi step để maintain orthogonality
    
    Complexity: O(K × n² × max_iter)
    
    Args:
        A: np.ndarray shape (n, n) - symmetric matrix
        k: Số eigenpairs cần tìm
        max_iter: Iterations tối đa cho mỗi eigenvector
        tol: Convergence tolerance
        verbose: Hiển thị progress bar
    
    Returns:
        tuple: (eigenvalues, eigenvectors)
            - eigenvalues: np.ndarray shape (k,), sorted descending
            - eigenvectors: np.ndarray shape (k, n), rows are eigenvectors
    """
    n = A.shape[0]
    k = min(k, n)
    
    # Use float64 for eigendecomposition (numerical stability)
    # but this is a small matrix (d x d), not the data matrix
    eigenvalues = np.zeros(k, dtype=np.float64)
    eigenvectors = np.zeros((k, n), dtype=np.float64)
    
    A_deflated = A.astype(np.float64, copy=True)
    
    iterator = range(k)
    if verbose:
        iterator = tqdm(iterator, desc="  Power Iteration (eigenvectors)")
    
    rng = np.random.RandomState(42)
    
    for i in iterator:
        # Random initial vector, orthogonal to previous eigenvectors
        v0 = rng.randn(n)
        if i > 0:
            v0 = orthogonalize_against(v0, eigenvectors[:i])
        
        # Power iteration on deflated matrix
        eigenvalue, eigenvector = power_iteration(A_deflated, max_iter, tol, v0)
        
        # Re-orthogonalize against all previous eigenvectors
        if i > 0:
            eigenvector = orthogonalize_against(eigenvector, eigenvectors[:i])
            # Recompute eigenvalue after re-orthogonalization
            eigenvalue = eigenvector @ A @ eigenvector
        
        eigenvalues[i] = eigenvalue
        eigenvectors[i] = eigenvector
        
        # Deflation: A' = A - λᵢ * vᵢ @ vᵢᵀ
        A_deflated = A_deflated - eigenvalue * np.outer(eigenvector, eigenvector)
    
    return eigenvalues, eigenvectors


# ==============================================================================
# 4. TRUNCATED SVD FROM SCRATCH
# ==============================================================================

def svd_truncated(X, k, max_iter=200, tol=1e-10, verbose=True):
    """
    Truncated SVD from scratch.
    
    SVD: X = U Σ Vᵀ
    
    Mối liên hệ SVD ↔ Eigendecomposition:
        - XᵀX = V Σ² Vᵀ  → V chứa right singular vectors
                            → eigenvalues của XᵀX = σᵢ²
        - XXᵀ = U Σ² Uᵀ  → U chứa left singular vectors
    
    Thuật toán:
        1. Tính ma trận C = XᵀX / n (proportional to covariance)
        2. Eigendecompose C → (eigenvalues, V)
        3. Singular values: σᵢ = √(n × λᵢ)
        4. Left singular vectors: uᵢ = X @ vᵢ / σᵢ
    
    Trick khi d < n (số chiều < số mẫu):
        - XᵀX là d×d (nhỏ hơn XXᵀ = n×n)
        - Tính toán hiệu quả hơn nhiều
    
    Args:
        X: np.ndarray shape (n, d) - data matrix (n samples, d features)
        k: Số singular components
        max_iter: Max iterations cho power iteration
        tol: Convergence tolerance
        verbose: Hiển thị progress
    
    Returns:
        tuple: (U, sigma, Vt)
            - U: np.ndarray shape (n, k) - left singular vectors
            - sigma: np.ndarray shape (k,) - singular values
            - Vt: np.ndarray shape (k, d) - right singular vectors (rows)
    """
    n, d = X.shape
    k = min(k, min(n, d))
    
    if verbose:
        print(f"  SVD: matrix {n}×{d}, computing top-{k} components")
    
    # Tính XᵀX (d×d) - hiệu quả hơn XXᵀ (n×n) khi d < n
    if verbose:
        print(f"  Computing XᵀX ({d}×{d})...")
    
    C = (X.T @ X) / n  # d×d matrix
    
    # Eigendecompose C
    eigenvalues, eigenvectors = eigendecompose_top_k(C, k, max_iter, tol, verbose)
    
    # Vt = eigenvectors (right singular vectors, shape k×d)
    Vt = eigenvectors
    
    # Singular values: σᵢ = √(n × λᵢ)
    # Clamp negative eigenvalues (numerical noise) to small positive
    eigenvalues = np.maximum(eigenvalues, 0)
    sigma = np.sqrt(n * eigenvalues)
    
    # Left singular vectors: U = X @ V / σ
    # U[:, i] = X @ Vt[i] / sigma[i]
    U = np.zeros((n, k), dtype=np.float64)
    for i in range(k):
        if sigma[i] > 1e-12:
            U[:, i] = (X @ Vt[i]) / sigma[i]
        else:
            U[:, i] = 0
    
    return U, sigma, Vt


# ==============================================================================
# 5. PCA CLASS (FROM SCRATCH)
# ==============================================================================

class PCA_Scratch:
    """
    Principal Component Analysis - implement từ đầu.
    
    ╔══════════════════════════════════════════════════════════════════╗
    ║  FROM SCRATCH IMPLEMENTATION                                    ║
    ║  KHÔNG dùng sklearn.decomposition.PCA                           ║
    ║  KHÔNG dùng numpy.linalg.svd                                    ║
    ║  Dùng Power Iteration + Deflation cho eigendecomposition        ║
    ╚══════════════════════════════════════════════════════════════════╝
    
    PCA Algorithm:
        1. Centering: X_centered = X - mean(X)
        2. Covariance: C = (1/n) Xᵀ_centered X_centered
        3. Eigendecompose C → top-K eigenvectors (principal components)
        4. Project: Z = X_centered @ Vᵀ (giảm chiều d → K)
        5. Reconstruct: X_hat = Z @ V + mean (tái tạo từ K components)
    
    Ý nghĩa hình học:
        - PCA tìm K hướng (principal components) mà data vary nhiều nhất
        - Projection lên K hướng này giữ maximum variance
        - Eigenvalue λᵢ = variance dọc theo principal component thứ i
        - Explained variance ratio = λᵢ / Σλ
    
    Attributes:
        n_components: Số principal components
        mean_: Mean vector (d,)
        components_: Principal components (k, d) - rows are components
        eigenvalues_: Eigenvalues (k,) - variance along each component
        explained_variance_ratio_: Tỷ lệ variance được giải thích
    """
    
    def __init__(self, n_components=100, max_iter=200, tol=1e-10):
        """
        Args:
            n_components: Số principal components cần giữ
            max_iter: Max iterations cho power iteration
            tol: Convergence tolerance
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        
        # Learned parameters (set sau khi fit)
        self.mean_ = None
        self.components_ = None     # Shape (k, d)
        self.eigenvalues_ = None    # Shape (k,)
        self.explained_variance_ratio_ = None
        self.n_features_ = None
    
    def fit(self, X, verbose=True):
        """
        Fit PCA model trên training data X.
        
        Args:
            X: np.ndarray shape (n, d) - n samples, d features
               Có thể là ảnh flatten: (n, H*W)
            verbose: Hiển thị progress
        
        Returns:
            self
        """
        n, d = X.shape
        self.n_features_ = d
        k = min(self.n_components, n, d)
        
        if verbose:
            print(f"\n  === PCA FROM SCRATCH ===")
            print(f"  Data: {n} samples × {d} features")
            print(f"  Components: {k}")
        
        # Step 1: Compute mean (cast to input dtype to avoid float32→float64 bloat)
        if verbose:
            print(f"  Step 1/3: Computing mean vector (dtype={X.dtype})...", flush=True)
        self.mean_ = np.mean(X, axis=0).astype(X.dtype)  # Shape (d,)
        
        # Step 2: Center data IN-PLACE (tiết kiệm bộ nhớ)
        if verbose:
            print(f"  Step 2/3: Centering data (X - mean) [in-place]...", flush=True)
        X_centered = X  # Reference, not copy
        X_centered -= self.mean_  # In-place subtraction → saves ~1.5GB RAM
        
        # Step 3: Compute covariance matrix and eigendecompose
        if verbose:
            print(f"  Step 3/3: Eigendecomposition via Power Iteration + Deflation...")
            if GPU_AVAILABLE:
                print(f"    [GPU] GPU acceleration ENABLED", flush=True)
        
        # Chọn chiến lược dựa trên kích thước ma trận
        if d <= n:
            # d ≤ n: Tính covariance matrix d×d trực tiếp
            # C = (1/n) Xᵀ_centered @ X_centered  (shape d×d)
            if verbose:
                print(f"    Strategy: d×d covariance ({d}×{d})")
            
            if GPU_AVAILABLE:
                # Keep covariance on GPU → eigendecompose on GPU (no round-trip)
                eigenvalues, eigenvectors = compute_and_decompose_gpu(
                    X_centered, k, self.max_iter, self.tol, verbose
                )
            else:
                C = (X_centered.T @ X_centered) / n
                eigenvalues, eigenvectors = eigendecompose_top_k(
                    C, k, self.max_iter, self.tol, verbose
                )
            
            self.components_ = eigenvectors  # Shape (k, d)
            self.eigenvalues_ = eigenvalues   # Shape (k,)
            
        else:
            # d > n: Dùng trick ma trận nhỏ n×n
            # L = (1/n) X_centered @ X_centeredᵀ  (shape n×n)
            # Eigenvectors của C = Xᵀ @ eigenvectors(L) (đã normalize)
            if verbose:
                print(f"    Strategy: n×n trick ({n}×{n})")
            
            L = (X_centered @ X_centered.T) / n  # Shape (n, n)
            
            if GPU_AVAILABLE:
                eigenvalues_L, eigenvectors_L = eigendecompose_top_k_gpu(
                    L, k, self.max_iter, self.tol, verbose
                )
            else:
                eigenvalues_L, eigenvectors_L = eigendecompose_top_k(
                    L, k, self.max_iter, self.tol, verbose
                )
            
            # Convert to d-dimensional eigenvectors
            # vᵢ = Xᵀ @ uᵢ / (√(n * λᵢ))
            components = np.zeros((k, d), dtype=np.float64)
            for i in range(k):
                if eigenvalues_L[i] > 1e-12:
                    v = X_centered.T @ eigenvectors_L[i]
                    v = v / np.linalg.norm(v)
                    components[i] = v
            
            self.components_ = components
            self.eigenvalues_ = eigenvalues_L
        
        # Compute explained variance ratio
        total_variance = np.sum(self.eigenvalues_)
        if total_variance > 1e-12:
            self.explained_variance_ratio_ = self.eigenvalues_ / total_variance
        else:
            self.explained_variance_ratio_ = np.zeros(k)
        
        if verbose:
            total_explained = np.sum(self.explained_variance_ratio_) * 100
            print(f"\n  PCA Training Complete!", flush=True)
            print(f"  Total variance explained: {total_explained:.2f}%", flush=True)
            print(f"  Top-5 eigenvalues: {self.eigenvalues_[:5]}", flush=True)
        
        return self
    
    def transform(self, X, whiten=True, l2_normalize=True):
        """
        Project data X onto principal components (giảm chiều).
        
        Công thức cơ bản: Z = (X - mean) @ Vᵀ
        
        Whitening (quan trọng cho face recognition!):
            Z_whitened = Z / √(eigenvalues)
            → Chuẩn hóa variance dọc mỗi PC thành 1
            → Tránh top PCs (lighting, pose) dominate distance
            → Giúp distance metrics đo IDENTITY thay vì variation
        
        L2 Normalization:
            Z_norm = Z / ‖Z‖₂
            → Feature vectors nằm trên unit hypersphere
            → Cosine distance = Euclidean distance (trên sphere)
            → Chuẩn cho face verification
        
        Args:
            X: np.ndarray shape (n, d) hoặc (d,)
            whiten: Áp dụng PCA whitening (chia sqrt eigenvalue)
            l2_normalize: L2 normalize feature vectors
        
        Returns:
            np.ndarray: shape (n, k) hoặc (k,) - projected features
        """
        single = X.ndim == 1
        if single:
            X = X[np.newaxis, :]
        
        # Project: Z = (X - mean) @ Vᵀ
        Z = (X - self.mean_) @ self.components_.T
        
        # Whitening: Z_w = Z / √(λ)
        if whiten and self.eigenvalues_ is not None:
            # Chỉ whiten các components có eigenvalue > 0
            scale = np.sqrt(np.maximum(self.eigenvalues_, 1e-12))
            Z = Z / scale
        
        # L2 normalization: Z_norm = Z / ‖Z‖
        if l2_normalize:
            norms = np.linalg.norm(Z, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)  # Avoid division by zero
            Z = Z / norms
        
        if single:
            return Z[0]
        return Z
    
    def inverse_transform(self, Z):
        """
        Reconstruct data từ projected features (tái tạo).
        
        Công thức: X_hat = Z @ V + mean
        (Lưu ý: chỉ hoạt động chính xác với raw projection, không whiten/L2)
        
        Args:
            Z: np.ndarray shape (n, k) - projected features (raw, non-whitened)
        
        Returns:
            np.ndarray: shape (n, d) - reconstructed data
        """
        if Z.ndim == 1:
            return Z @ self.components_ + self.mean_
        return Z @ self.components_ + self.mean_
    
    def transform_raw(self, X):
        """
        Raw projection without whitening/normalization (cho reconstruction).
        """
        single = X.ndim == 1
        if single:
            X = X[np.newaxis, :]
        Z = (X - self.mean_) @ self.components_.T
        if single:
            return Z[0]
        return Z
    
    def fit_transform(self, X, verbose=True):
        """
        Fit và transform trong một bước.
        """
        self.fit(X, verbose)
        return self.transform(X)


# ==============================================================================
# 6. GPU-ACCELERATED EIGENDECOMPOSITION
# ==============================================================================

def eigendecompose_top_k_gpu(A_np, k, max_iter=200, tol=1e-10, verbose=True):
    """
    GPU-accelerated version of eigendecompose_top_k using CuPy.
    
    Transfers covariance matrix to GPU and runs all Power Iteration + Deflation
    on GPU. Only transfers final results back to CPU.
    
    This is much faster for large matrices because:
        - Matrix-vector multiply (A @ v) runs on GPU CUDA cores
        - Outer product (v @ v.T) for deflation also on GPU
        - No CPU-GPU transfer during iteration loops
    
    Args:
        A_np: np.ndarray shape (n, n) - symmetric matrix (on CPU)
        k: Number of eigenpairs
        max_iter: Max iterations per eigenvector
        tol: Convergence tolerance
        verbose: Show progress
    
    Returns:
        tuple: (eigenvalues, eigenvectors) as numpy arrays on CPU
    """
    n = A_np.shape[0]
    k = min(k, n)
    
    if verbose:
        mem_mb = n * n * 8 / 1e6
        print(f"    [GPU] Transferring {n}×{n} matrix to GPU ({mem_mb:.0f} MB)...", flush=True)
    
    # Transfer to GPU as float64 for numerical stability (covariance is small: d×d)
    A_gpu = cp.asarray(A_np, dtype=cp.float64)
    A_deflated = A_gpu.copy()
    
    eigenvalues = np.zeros(k, dtype=np.float64)
    eigenvectors_gpu = cp.zeros((k, n), dtype=cp.float64)
    
    rng_np = np.random.RandomState(42)
    
    iterator = range(k)
    if verbose:
        iterator = tqdm(iterator, desc="  Power Iteration [GPU]")
    
    for i in iterator:
        # Random initial vector on GPU
        v0_np = rng_np.randn(n)
        v = cp.asarray(v0_np, dtype=cp.float64)
        
        # Orthogonalize against previous eigenvectors (on GPU)
        if i > 0:
            for j in range(i):
                proj = cp.dot(v, eigenvectors_gpu[j])
                v = v - proj * eigenvectors_gpu[j]
        
        norm = cp.linalg.norm(v)
        if norm > 1e-12:
            v = v / norm
        
        # Power iteration on GPU
        eigenvalue_gpu = cp.float64(0.0)
        for iteration in range(max_iter):
            w = A_deflated @ v  # GPU matrix-vector multiply
            eigenvalue_gpu = cp.dot(v, w)
            
            norm_w = cp.linalg.norm(w)
            if norm_w < 1e-15:
                break
            v_new = w / norm_w
            
            diff = min(float(cp.linalg.norm(v_new - v)), 
                       float(cp.linalg.norm(v_new + v)))
            v = v_new
            
            if diff < tol:
                break
        
        # Re-orthogonalize
        if i > 0:
            for j in range(i):
                proj = cp.dot(v, eigenvectors_gpu[j])
                v = v - proj * eigenvectors_gpu[j]
            norm = cp.linalg.norm(v)
            if norm > 1e-12:
                v = v / norm
            eigenvalue_gpu = v @ A_gpu @ v
        
        eigenvalues[i] = float(eigenvalue_gpu)
        eigenvectors_gpu[i] = v
        
        # Deflation on GPU
        A_deflated = A_deflated - eigenvalue_gpu * cp.outer(v, v)
    
    # Transfer eigenvectors back to CPU
    eigenvectors = cp.asnumpy(eigenvectors_gpu)
    
    # Free GPU memory
    del A_gpu, A_deflated, eigenvectors_gpu
    cp.get_default_memory_pool().free_all_blocks()
    
    if verbose:
        print(f"    [GPU] Eigendecomposition complete!", flush=True)
    
    return eigenvalues, eigenvectors


def compute_and_decompose_gpu(X, k, max_iter=200, tol=1e-10, verbose=True):
    """
    Compute covariance AND eigendecompose entirely on GPU.
    
    This avoids the wasteful GPU→CPU→GPU round-trip that happens when
    compute_covariance_gpu and eigendecompose_top_k_gpu are called separately.
    
    Pipeline (all on GPU):
        1. Compute C = X.T @ X / n  (batched, supports float32 input)
        2. Eigendecompose C → top-K eigenpairs (Power Iteration on GPU)
        3. Transfer only final eigenvectors (small) back to CPU
    
    Memory usage:
        - X batches: ~256 MB per batch (transferred then freed)
        - C matrix (d×d, float64): ~328 MB for d=6400
        - Eigenvectors (k×d, float64): ~15 MB for k=300, d=6400
        - Total GPU peak: ~600 MB — fits easily in T4's 15 GB
    
    Args:
        X: np.ndarray shape (n, d) - centered data matrix (on CPU, float32 or float64)
        k: Number of eigenpairs
        max_iter: Max iterations per eigenvector
        tol: Convergence tolerance
        verbose: Show progress
    
    Returns:
        tuple: (eigenvalues, eigenvectors) as numpy arrays on CPU
    """
    n, d = X.shape
    itemsize = X.dtype.itemsize  # 4 for float32, 8 for float64
    
    free_mem, total_mem = cp.cuda.Device(0).mem_info
    free_gb = free_mem / 1e9
    data_gb = n * d * itemsize / 1e9
    
    if verbose:
        print(f"    [GPU] Covariance + Eigendecompose (all on GPU)", flush=True)
        print(f"    [GPU] Data: {n:,}×{d} ({X.dtype}), {data_gb:.2f} GB", flush=True)
        print(f"    [GPU] VRAM: {free_gb:.1f} / {total_mem/1e9:.1f} GB free", flush=True)
    
    # Step 1: Compute covariance on GPU (batched to fit in VRAM)
    # Use float64 for covariance accumulation (numerical stability)
    # Leave 1 GB buffer for eigendecomposition
    max_batch_bytes = int((free_gb - 1.0) * 1e9 * 0.5)
    max_batch_rows = max(1000, max_batch_bytes // (d * itemsize))
    
    if verbose:
        n_batches = (n + max_batch_rows - 1) // max_batch_rows
        print(f"    [GPU] Computing covariance in {n_batches} batch(es) "
              f"({max_batch_rows:,} rows/batch)...", flush=True)
    
    C_gpu = cp.zeros((d, d), dtype=cp.float64)
    
    for start in range(0, n, max_batch_rows):
        end = min(start + max_batch_rows, n)
        # Transfer batch to GPU (auto-converts float32→float64 in matmul)
        X_batch_gpu = cp.asarray(X[start:end])
        C_gpu += X_batch_gpu.T @ X_batch_gpu
        del X_batch_gpu
        cp.get_default_memory_pool().free_all_blocks()
    
    C_gpu /= n
    
    if verbose:
        cov_mb = d * d * 8 / 1e6
        print(f"    [GPU] Covariance {d}×{d} on GPU ({cov_mb:.0f} MB)", flush=True)
    
    # Step 2: Eigendecompose on GPU (C_gpu stays on GPU — no round-trip!)
    k = min(k, d)
    A_gpu = C_gpu  # Already on GPU
    A_deflated = A_gpu.copy()
    
    eigenvalues = np.zeros(k, dtype=np.float64)
    eigenvectors_gpu = cp.zeros((k, d), dtype=cp.float64)
    
    rng_np = np.random.RandomState(42)
    
    iterator = range(k)
    if verbose:
        iterator = tqdm(iterator, desc="  Power Iteration [GPU]")
    
    for i in iterator:
        v0_np = rng_np.randn(d)
        v = cp.asarray(v0_np, dtype=cp.float64)
        
        # Orthogonalize against previous eigenvectors
        if i > 0:
            for j in range(i):
                proj = cp.dot(v, eigenvectors_gpu[j])
                v = v - proj * eigenvectors_gpu[j]
        
        norm = cp.linalg.norm(v)
        if norm > 1e-12:
            v = v / norm
        
        eigenvalue_gpu = cp.float64(0.0)
        for iteration in range(max_iter):
            w = A_deflated @ v
            eigenvalue_gpu = cp.dot(v, w)
            
            norm_w = cp.linalg.norm(w)
            if norm_w < 1e-15:
                break
            v_new = w / norm_w
            
            diff = min(float(cp.linalg.norm(v_new - v)),
                       float(cp.linalg.norm(v_new + v)))
            v = v_new
            
            if diff < tol:
                break
        
        # Re-orthogonalize
        if i > 0:
            for j in range(i):
                proj = cp.dot(v, eigenvectors_gpu[j])
                v = v - proj * eigenvectors_gpu[j]
            norm = cp.linalg.norm(v)
            if norm > 1e-12:
                v = v / norm
            eigenvalue_gpu = v @ A_gpu @ v
        
        eigenvalues[i] = float(eigenvalue_gpu)
        eigenvectors_gpu[i] = v
        
        A_deflated = A_deflated - eigenvalue_gpu * cp.outer(v, v)
    
    # Step 3: Transfer only small result arrays back to CPU
    eigenvectors = cp.asnumpy(eigenvectors_gpu)
    
    del A_gpu, C_gpu, A_deflated, eigenvectors_gpu
    cp.get_default_memory_pool().free_all_blocks()
    
    if verbose:
        free_after, _ = cp.cuda.Device(0).mem_info
        print(f"    [GPU] Done! VRAM freed: {free_after/1e9:.1f} GB free", flush=True)
    
    return eigenvalues, eigenvectors


def compute_covariance_gpu(X, verbose=True):
    """
    Compute covariance matrix X.T @ X / n using batched GPU operations.
    
    For large X that doesn't fit entirely on GPU, processes in batches:
        C = (1/n) * Σ_batches X_batch.T @ X_batch
    
    Args:
        X: np.ndarray shape (n, d) - centered data matrix (on CPU, float32 or float64)
        verbose: Show progress
    
    Returns:
        np.ndarray shape (d, d) - covariance matrix (on CPU, float64)
    """
    n, d = X.shape
    itemsize = X.dtype.itemsize
    
    # Estimate GPU memory needed for full transfer
    data_size_gb = n * d * itemsize / 1e9
    free_mem, _ = cp.cuda.Device(0).mem_info
    free_gb = free_mem / 1e9
    
    if verbose:
        print(f"    [GPU] Computing covariance: data={data_size_gb:.2f}GB ({X.dtype}), "
              f"GPU free={free_gb:.2f}GB", flush=True)
    
    # Use batched computation if data is large
    # Leave 0.5 GB buffer for other GPU operations
    max_batch_bytes = int((free_gb - 0.5) * 1e9 * 0.6)
    max_batch_rows = max(1000, max_batch_bytes // (d * itemsize))
    
    # Accumulate in float64 for numerical stability
    C_gpu = cp.zeros((d, d), dtype=cp.float64)
    
    for start in range(0, n, max_batch_rows):
        end = min(start + max_batch_rows, n)
        X_batch_gpu = cp.asarray(X[start:end])
        C_gpu += X_batch_gpu.T @ X_batch_gpu
        del X_batch_gpu
    
    C_gpu /= n
    C_np = cp.asnumpy(C_gpu)
    
    del C_gpu
    cp.get_default_memory_pool().free_all_blocks()
    
    return C_np


# ==============================================================================
# 7. INCREMENTAL COVARIANCE (CHO DATASETS LỚN)
# ==============================================================================

def compute_mean_incremental(data_generator, n_features):
    """
    Tính mean vector incrementally (không cần load toàn bộ data vào RAM).
    
    Welford's online algorithm:
        mean_new = mean_old + (x - mean_old) / count
    
    Args:
        data_generator: Generator yield batches of shape (batch_size, n_features)
        n_features: Dimension of each sample
    
    Returns:
        tuple: (mean, count)
    """
    mean = np.zeros(n_features, dtype=np.float64)
    count = 0
    
    for batch in data_generator:
        batch_size = batch.shape[0]
        batch_mean = np.mean(batch, axis=0)
        
        # Update running mean
        total = count + batch_size
        mean = (mean * count + batch_mean * batch_size) / total
        count = total
    
    return mean, count


def compute_covariance_incremental(data_generator, mean, n_features):
    """
    Tính covariance matrix incrementally.
    
    C = (1/n) Σᵢ (xᵢ - mean)(xᵢ - mean)ᵀ
      = (1/n) Σ_batches [X_batch - mean]ᵀ @ [X_batch - mean]
    
    Args:
        data_generator: Generator yield batches
        mean: Pre-computed mean vector
        n_features: Feature dimension
    
    Returns:
        tuple: (covariance_matrix, count)
    """
    C = np.zeros((n_features, n_features), dtype=np.float64)
    count = 0
    
    for batch in data_generator:
        batch_centered = batch - mean
        C += batch_centered.T @ batch_centered
        count += batch.shape[0]
    
    C /= count
    return C, count
