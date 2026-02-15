# Enhanced Eigenfaces: PCA/SVD for Face Recognition

**Mathematics for AI — Final Project**  
Classical Eigenfaces improved with multi-scale features, illumination normalization, and ensemble metrics. All algorithms implemented **from scratch** (no `sklearn.decomposition.PCA`, no dlib).

---

## Contributors

- [*Bùi Trần Tấn Phát*](https://github.com/FlynnBui399) — *Ho Chi Minh City University of Technology and Engineering (HCMUTE)* 
- [*Nguyễn Nhật Phát*](https://github.com/PhoenixEvo) — *Ho Chi Minh City University of Technology and Engineering (HCMUTE)* 

---

## Features

- **Baseline:** Standard Eigenfaces (Turk & Pentland 1991) — PCA, single scale, cosine distance.
- **Enhanced:**  
  - **C1** Multi-scale pyramid (`[80, 40]`).  
  - **C2** Ensemble distance metrics (optional).  
  - **C3** Illumination normalization (Histogram Equalization / Tan–Triggs).
- **Evaluation:** LFW, CFP-FP, AgeDB-30 (10-fold verification, AUC, EER).
- **Ablation:** Sweep components K, multi-scale configs; tables and figures for report.

---

## Project structure

```
PRJ_MAAI/
├── src/
│   ├── config.py          # Paths & hyperparameters
│   ├── data_loader.py     # CASIA-WebFace (RecordIO) + LFW/CFP-FP/AgeDB (.bin)
│   ├── preprocessing.py  # Histogram eq, Tan–Triggs, standardize (from scratch)
│   ├── linalg_scratch.py  # PCA, Power Iteration, Gram–Schmidt, GPU (CuPy)
│   ├── eigenfaces.py     # Baseline, MultiScale, Enhanced
│   ├── metrics.py        # Euclidean, cosine, Manhattan, Chi-square, ensemble
│   ├── evaluation.py     # K-fold, ROC, AUC, EER, threshold
│   └── visualization.py  # Eigenfaces, ROC, ablation plots
├── run_experiment.py      # Full pipeline: train → evaluate → save model
├── run_demo.py            # Interactive demo: register faces, 1:N identify, 1:1 verify
├── run_ablation_extra.py # Optional ablation: sweep K, multi-scale configs
├── Enhanced_Eigenfaces_Colab.ipynb  # Same pipeline on Google Colab (GPU)
├── requirements.txt
├── ablation_plan.txt      # Ablation design (Exp0–Exp6, section 5)
└── results/
    ├── figures/          # Plots (eigenfaces, ROC, ablation)
    ├── tables/           # CSV (results_table, ablation_5_1, ablation_5_2)
    └── trained_models.npz  # Saved model (not in repo; train or download)
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/FlynnBui399/Face-Verification-with-Enhanced-Eigenfaces.git
cd PRJ_MAAI
pip install -r requirements.txt
```

Optional (GPU):

```bash
pip install cupy-cuda12x   # requires CUDA Toolkit
```

### 2. Dataset

Datasets are **not** included in the repo (large size). You can download it at: https://www.kaggle.com/datasets/debarghamitraroy/casia-webface You need:

| Data        | Path under `dataset/`     | Description                    |
|-------------|---------------------------|--------------------------------|
| CASIA-WebFace | `casia-webface/` (train.rec, train.idx, property) | Training faces |
| LFW         | `eval/lfw.bin`            | Verification pairs             |
| CFP-FP      | `eval/cfp_fp.bin`         | Frontal–profile                |
| AgeDB-30    | `eval/agedb_30.bin`       | Age variation                  |

Create the layout and place your files:

```
dataset/
├── casia-webface/
│   ├── train.rec
│   ├── train.idx
│   └── property
└── eval/
    ├── lfw.bin
    ├── cfp_fp.bin
    └── agedb_30.bin
```

---

## Usage

### Train and evaluate (full pipeline)

```bash
python -u run_experiment.py
```

- Trains Baseline and Enhanced (HistEq + MultiScale).
- Evaluates on LFW, CFP-FP, AgeDB-30.
- Saves figures and tables under `results/`.
- Saves **trained model** to `results/trained_models.npz`.

### Demo (after training)

```bash
python run_demo.py
```

- Loads `results/trained_models.npz`.
- Menu: register faces → 1:N identification → 1:1 verification (paths to images).

### Optional ablation (section 5)

```bash
python run_ablation_extra.py
```

- **5.1** Baseline LFW: K ∈ {100, 200, 300, 400}.
- **5.2** Multi-scale: [80], [80, 40], [80, 40, 20].
- Writes `results/tables/ablation_5_1_*.csv`, `ablation_5_2_*.csv` and figures.

---

## Results (example)

| Method   | LFW (Acc%) | CFP-FP (Acc%) | AgeDB-30 (Acc%) |
|----------|------------|---------------|-----------------|
| Baseline | 69.65      | 56.20         | 55.90           |
| Enhanced | 70.55      | 58.11         | 58.70           |

(Exact numbers depend on config and data; run `run_experiment.py` to reproduce.)

---

## Config

Edit `src/config.py` for:

- `IMAGE_SIZE`, `N_TRAIN_SAMPLES`, `N_COMPONENTS`
- `MULTI_SCALE_SIZES`, `USE_GPU`
- Paths to `dataset/` if different

---

## License

This project is for academic use (Mathematics for AI course). Dataset usage must comply with CASIA-WebFace, LFW, CFP-FP, and AgeDB-30 terms.
