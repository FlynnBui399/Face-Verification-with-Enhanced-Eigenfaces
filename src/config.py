"""
Configuration parameters for Enhanced Eigenfaces project.
All hyperparameters and paths are defined here for easy modification.
"""

import os

# ==============================================================================
# PATHS
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
CACHE_DIR = os.path.join(DATASET_DIR, "cache")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")

# CASIA-WebFace paths
CASIA_DIR = os.path.join(DATASET_DIR, "casia-webface")
CASIA_REC = os.path.join(CASIA_DIR, "train.rec")
CASIA_IDX = os.path.join(CASIA_DIR, "train.idx")
CASIA_LST = os.path.join(CASIA_DIR, "train.lst")
CASIA_PROPERTY = os.path.join(CASIA_DIR, "property")

# Eval dataset paths
EVAL_DIR = os.path.join(DATASET_DIR, "eval")
LFW_BIN = os.path.join(EVAL_DIR, "lfw.bin")
CFP_FP_BIN = os.path.join(EVAL_DIR, "cfp_fp.bin")
AGEDB_BIN = os.path.join(EVAL_DIR, "agedb_30.bin")

# ==============================================================================
# IMAGE PARAMETERS
# ==============================================================================
ORIGINAL_SIZE = 112          # Original image size from CASIA-WebFace
IMAGE_SIZE = 64              # Resize target for PCA (64x64 grayscale)
MULTI_SCALE_SIZES = [64, 32]      # Multi-scale pyramid sizes (removed 16 - too small)

# ==============================================================================
# PCA PARAMETERS
# ==============================================================================
N_TRAIN_SAMPLES = 40000      # Number of training images for PCA (memory-friendly)
N_COMPONENTS = 200            # Number of principal components to keep
POWER_ITER_MAX = 100          # Max iterations for power iteration
POWER_ITER_TOL = 1e-8         # Convergence tolerance

# ==============================================================================
# ENSEMBLE METRIC PARAMETERS
# ==============================================================================
METRIC_NAMES = ["euclidean", "cosine", "manhattan", "chi_square"]
# Initial weights (will be optimized via grid search)
METRIC_WEIGHTS = [0.3, 0.3, 0.2, 0.2]

# ==============================================================================
# EVALUATION PARAMETERS
# ==============================================================================
N_FOLDS = 10                  # K-fold cross-validation
THRESHOLD_STEPS = 200         # Number of threshold steps for ROC

# ==============================================================================
# HARDWARE PARAMETERS
# ==============================================================================
BATCH_SIZE = 5000             # Batch size for data loading
USE_GPU = True               # Disabled: CUDA toolkit not fully installed on this system

# ==============================================================================
# RANDOM SEED
# ==============================================================================
RANDOM_SEED = 42
