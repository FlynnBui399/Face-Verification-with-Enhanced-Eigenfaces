"""
Image Preprocessing - Tiền xử lý ảnh khuôn mặt (FROM SCRATCH).

Tất cả các thuật toán preprocessing được implement từ đầu sử dụng
numpy cơ bản, KHÔNG dùng OpenCV hay sklearn.

Các kỹ thuật:
  1. Histogram Equalization - Cân bằng histogram
  2. Gaussian Blur - Làm mờ Gauss (2D convolution from scratch)
  3. Difference of Gaussians (DoG) - Lọc tần số
  4. Tan-Triggs Normalization - Chuẩn hóa ánh sáng robust
  5. Full Illumination Normalization Pipeline

Toán học:
  - Histogram Equalization: T(r) = (L-1) * CDF(r)
  - Gaussian: G(x,y) = (1/2πσ²) * exp(-(x²+y²)/2σ²)
  - DoG: DoG(x,y) = G(x,y,σ₁) - G(x,y,σ₂)
  - Tan-Triggs: γ correction → DoG → contrast equalization

Author: Mathematics for AI - Final Project
"""

import numpy as np


# ==============================================================================
# 1. HISTOGRAM EQUALIZATION (FROM SCRATCH)
# ==============================================================================

def histogram_equalization(image):
    """
    Cân bằng histogram - tăng contrast cho ảnh grayscale.
    
    Thuật toán (implement from scratch):
        1. Tính histogram H(k) = số pixel có giá trị k, k ∈ [0, 255]
        2. Tính CDF (Cumulative Distribution Function):
           CDF(k) = Σ_{i=0}^{k} H(i) / N
        3. Áp dụng transform: T(r) = round(255 * CDF(r))
    
    Ý nghĩa toán học:
        - CDF map phân phối bất kỳ → phân phối đều [0, 255]
        - Tăng contrast ở vùng histogram dày, giảm ở vùng thưa
    
    Args:
        image: np.ndarray shape (H, W), giá trị [0, 255] hoặc [0.0, 1.0]
    
    Returns:
        np.ndarray: Ảnh sau equalization, cùng shape, giá trị [0, 255]
    """
    # Normalize to [0, 255] range
    img = image.copy()
    if img.max() <= 1.0:
        img = img * 255.0
    img = np.clip(img, 0, 255).astype(np.int32)
    
    H, W = img.shape
    N = H * W
    L = 256  # Number of intensity levels
    
    # Step 1: Compute histogram
    # H[k] = number of pixels with intensity k
    hist = np.zeros(L, dtype=np.int64)
    for k in range(L):
        hist[k] = np.sum(img == k)
    
    # Step 2: Compute CDF (cumulative distribution function)
    # CDF[k] = Σ_{i=0}^{k} hist[i] / N
    cdf = np.cumsum(hist).astype(np.float64) / N
    
    # Step 3: Apply transform
    # T(r) = round(255 * CDF(r))
    lut = np.round(255.0 * cdf).astype(np.int32)  # Lookup table
    
    result = lut[img]
    
    return result.astype(np.float32)


# ==============================================================================
# 2. GAUSSIAN BLUR (FROM SCRATCH - 2D Convolution)
# ==============================================================================

def create_gaussian_kernel(size, sigma):
    """
    Tạo Gaussian kernel 2D.
    
    Công thức Gaussian 2D:
        G(x, y) = (1 / 2πσ²) * exp(-(x² + y²) / 2σ²)
    
    Args:
        size: Kích thước kernel (phải lẻ, ví dụ 3, 5, 7)
        sigma: Độ lệch chuẩn (standard deviation)
    
    Returns:
        np.ndarray: Gaussian kernel, shape (size, size), tổng = 1
    """
    # Tạo grid tọa độ centered tại (0, 0)
    half = size // 2
    x = np.arange(-half, half + 1, dtype=np.float64)
    y = np.arange(-half, half + 1, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)
    
    # Áp dụng công thức Gaussian
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    
    # Normalize để tổng = 1 (conservation of energy)
    kernel = kernel / np.sum(kernel)
    
    return kernel


def convolve2d(image, kernel):
    """
    2D Convolution from scratch.
    
    Công thức convolution:
        (f * g)[m, n] = Σ_i Σ_j f[m-i, n-j] * g[i, j]
    
    Implementation: sử dụng zero-padding để giữ kích thước output = input.
    
    Tối ưu: Thay vì nested loops, dùng numpy vectorization với sliding window.
    
    Args:
        image: np.ndarray shape (H, W)
        kernel: np.ndarray shape (kH, kW) - phải có kích thước lẻ
    
    Returns:
        np.ndarray: Kết quả convolution, cùng shape (H, W)
    """
    H, W = image.shape
    kH, kW = kernel.shape
    pad_h = kH // 2
    pad_w = kW // 2
    
    # Zero-padding
    padded = np.zeros((H + 2*pad_h, W + 2*pad_w), dtype=np.float64)
    padded[pad_h:pad_h+H, pad_w:pad_w+W] = image
    
    # Convolution sử dụng vectorized sliding window
    # Thay vì O(H*W*kH*kW) nested loops, dùng numpy broadcasting
    output = np.zeros((H, W), dtype=np.float64)
    
    for i in range(kH):
        for j in range(kW):
            output += padded[i:i+H, j:j+W] * kernel[i, j]
    
    return output


def gaussian_blur(image, sigma=1.0):
    """
    Áp dụng Gaussian blur cho ảnh.
    
    Kernel size được tính tự động: size = 2 * ceil(3*sigma) + 1
    (bao phủ 99.7% của Gaussian distribution theo quy tắc 3-sigma)
    
    Args:
        image: np.ndarray shape (H, W)
        sigma: Standard deviation của Gaussian
    
    Returns:
        np.ndarray: Ảnh sau blur
    """
    # Tự động tính kernel size
    size = 2 * int(np.ceil(3 * sigma)) + 1
    size = max(size, 3)  # Minimum 3x3
    
    kernel = create_gaussian_kernel(size, sigma)
    return convolve2d(image, kernel)


# ==============================================================================
# 3. DIFFERENCE OF GAUSSIANS - DoG (FROM SCRATCH)
# ==============================================================================

def difference_of_gaussians(image, sigma1=1.0, sigma2=2.0):
    """
    Difference of Gaussians (DoG) filter.
    
    Công thức:
        DoG(x, y) = G(x, y, σ₁) - G(x, y, σ₂)
    
    Ý nghĩa:
        - DoG xấp xỉ Laplacian of Gaussian (LoG)
        - Loại bỏ low-frequency components (illumination variation)
        - Giữ lại mid-frequency components (facial features, edges)
        - σ₁ < σ₂: blur nhẹ - blur mạnh = band-pass filter
    
    Trong face recognition:
        - Loại bỏ ảnh hưởng ánh sáng không đều
        - Giữ lại cấu trúc khuôn mặt (mắt, mũi, miệng)
    
    Args:
        image: np.ndarray shape (H, W)
        sigma1: Sigma nhỏ (high-frequency pass)
        sigma2: Sigma lớn (low-frequency suppress)
    
    Returns:
        np.ndarray: Ảnh sau DoG filter
    """
    blur1 = gaussian_blur(image, sigma1)
    blur2 = gaussian_blur(image, sigma2)
    
    dog = blur1 - blur2
    
    return dog


# ==============================================================================
# 4. TAN-TRIGGS NORMALIZATION (FROM SCRATCH)
# ==============================================================================

def gamma_correction(image, gamma=0.2):
    """
    Gamma correction - nén dynamic range.
    
    Công thức: I_out = I_in^γ
    
    Với γ < 1: tăng sáng vùng tối (nén highlights, mở rộng shadows)
    → Giảm ảnh hưởng của ánh sáng cực mạnh/yếu
    
    Args:
        image: np.ndarray, giá trị >= 0
        gamma: Exponent (0 < γ < 1 cho face recognition)
    
    Returns:
        np.ndarray: Ảnh sau gamma correction
    """
    # Ensure non-negative
    img = np.abs(image) + 1e-10
    return np.power(img, gamma)


def contrast_equalization(image, alpha=0.1, tau=10.0):
    """
    Contrast equalization (Tan-Triggs 2010).
    
    Two-stage normalization:
        Stage 1: I' = I / (mean(|I|^α))^(1/α)
        Stage 2: I'' = I' / (mean(min(|I'|, τ)^α))^(1/α)
        Stage 3: I_final = τ * tanh(I'' / τ)
    
    Ý nghĩa:
        - Stage 1: Normalize theo global contrast
        - Stage 2: Robust normalization (clamp outliers tại τ)
        - Stage 3: Compress dynamic range vào [-τ, τ]
    
    Args:
        image: np.ndarray shape (H, W)
        alpha: Exponent cho normalization (mặc định 0.1)
        tau: Truncation threshold
    
    Returns:
        np.ndarray: Ảnh sau contrast equalization
    """
    img = image.copy()
    
    # Stage 1: Global contrast normalization
    mean_abs_alpha = np.mean(np.abs(img) ** alpha)
    if mean_abs_alpha > 1e-10:
        img = img / (mean_abs_alpha ** (1.0 / alpha))
    
    # Stage 2: Robust normalization
    clipped = np.minimum(np.abs(img), tau)
    mean_clipped = np.mean(clipped ** alpha)
    if mean_clipped > 1e-10:
        img = img / (mean_clipped ** (1.0 / alpha))
    
    # Stage 3: Compress to [-tau, tau] using tanh
    img = tau * np.tanh(img / tau)
    
    return img


def tan_triggs_normalization(image, gamma=0.2, sigma1=1.0, sigma2=2.0,
                              alpha=0.1, tau=10.0):
    """
    Tan-Triggs Illumination Normalization Pipeline.
    
    Reference: Tan & Triggs, "Enhanced Local Texture Feature Sets for
    Face Recognition Under Difficult Lighting Conditions", IEEE TIP, 2010.
    
    Pipeline:
        1. Gamma correction: nén dynamic range
        2. DoG filtering: loại bỏ low-frequency illumination
        3. Contrast equalization: normalize intensity distribution
    
    Đây là kỹ thuật chuẩn để giảm ảnh hưởng ánh sáng trong face recognition.
    
    Args:
        image: np.ndarray shape (H, W), giá trị [0, 255]
        gamma: Gamma correction exponent
        sigma1, sigma2: DoG parameters
        alpha: Contrast equalization exponent
        tau: Contrast equalization threshold
    
    Returns:
        np.ndarray: Ảnh đã normalize, giá trị trong [-tau, tau]
    """
    # Step 1: Gamma correction
    img = gamma_correction(image, gamma)
    
    # Step 2: DoG filtering
    img = difference_of_gaussians(img, sigma1, sigma2)
    
    # Step 3: Contrast equalization
    img = contrast_equalization(img, alpha, tau)
    
    return img


# ==============================================================================
# 5. FULL PREPROCESSING PIPELINE
# ==============================================================================

def normalize_to_unit(image):
    """
    Normalize ảnh về [0, 1] range.
    """
    img = image.copy()
    min_val = img.min()
    max_val = img.max()
    if max_val - min_val > 1e-10:
        img = (img - min_val) / (max_val - min_val)
    return img


def standardize(image):
    """
    Standardize ảnh: zero mean, unit variance.
    
    Công thức: z = (x - μ) / σ
    
    Đây là bước quan trọng trước PCA để đảm bảo các features
    có cùng scale.
    """
    mean = np.mean(image)
    std = np.std(image)
    if std > 1e-10:
        return (image - mean) / std
    return image - mean


def preprocess_image(image, use_hist_eq=False, use_tan_triggs=False):
    """
    Full preprocessing pipeline cho một ảnh.
    
    Args:
        image: np.ndarray shape (H, W), giá trị [0, 255]
        use_hist_eq: Sử dụng histogram equalization
        use_tan_triggs: Sử dụng Tan-Triggs normalization
    
    Returns:
        np.ndarray: Ảnh đã preprocess, flatten thành vector 1D
    """
    img = image.copy()
    
    if use_hist_eq:
        img = histogram_equalization(img)
    
    if use_tan_triggs:
        img = tan_triggs_normalization(img)
    
    # Standardize
    img = standardize(img)
    
    return img


def preprocess_batch(images, use_hist_eq=False, use_tan_triggs=False):
    """
    Preprocess một batch ảnh.
    
    Args:
        images: np.ndarray shape (N, H, W)
        use_hist_eq: Sử dụng histogram equalization
        use_tan_triggs: Sử dụng Tan-Triggs normalization
    
    Returns:
        np.ndarray: shape (N, H, W), đã preprocess
    """
    N = images.shape[0]
    result = np.zeros_like(images)
    
    for i in range(N):
        result[i] = preprocess_image(images[i], use_hist_eq, use_tan_triggs)
    
    return result


def resize_image(image, target_size):
    """
    Resize ảnh sử dụng bilinear interpolation (from scratch).
    
    Bilinear interpolation:
        f(x, y) = f(0,0)(1-x)(1-y) + f(1,0)x(1-y) + f(0,1)(1-x)y + f(1,1)xy
    
    Args:
        image: np.ndarray shape (H, W)
        target_size: int, kích thước output (target_size × target_size)
    
    Returns:
        np.ndarray: Ảnh đã resize, shape (target_size, target_size)
    """
    H, W = image.shape
    new_H, new_W = target_size, target_size
    
    # Tạo grid tọa độ trong ảnh gốc
    # Map pixel (i, j) trong output → (y, x) trong input
    row_ratio = H / new_H
    col_ratio = W / new_W
    
    output = np.zeros((new_H, new_W), dtype=np.float64)
    
    for i in range(new_H):
        for j in range(new_W):
            # Tọa độ thực trong ảnh gốc
            y = i * row_ratio
            x = j * col_ratio
            
            # 4 pixel lân cận
            y0 = int(np.floor(y))
            x0 = int(np.floor(x))
            y1 = min(y0 + 1, H - 1)
            x1 = min(x0 + 1, W - 1)
            
            # Trọng số
            wy = y - y0
            wx = x - x0
            
            # Bilinear interpolation
            output[i, j] = (image[y0, x0] * (1 - wy) * (1 - wx) +
                           image[y1, x0] * wy * (1 - wx) +
                           image[y0, x1] * (1 - wy) * wx +
                           image[y1, x1] * wy * wx)
    
    return output


def resize_batch(images, target_size):
    """
    Resize một batch ảnh.
    Sử dụng PIL cho tốc độ (resize không phải thuật toán cốt lõi).
    
    Args:
        images: np.ndarray shape (N, H, W)
        target_size: int
    
    Returns:
        np.ndarray: shape (N, target_size, target_size)
    """
    from PIL import Image as PILImage
    
    N = images.shape[0]
    out_dtype = images.dtype if images.dtype in (np.float32, np.float64) else np.float32
    result = np.zeros((N, target_size, target_size), dtype=out_dtype)
    
    for i in range(N):
        img = PILImage.fromarray(images[i].astype(np.uint8), mode='L')
        img_resized = img.resize((target_size, target_size), PILImage.BILINEAR)
        result[i] = np.array(img_resized, dtype=out_dtype)
    
    return result
