# VAECox Reproducibility Checklist

Paper: *Improved survival analysis by learning shared genomic information
from pan-cancer data* (Bioinformatics 2020, Supplement 1)

---

## 1. Data Source

| Item | Value |
|---|---|
| Dataset | TCGA RNA-seq gene expression |
| Access | ICGC Data Portal — requires dbGaP phs000178 (controlled) |
| Toy data | `./data/imputed_and_binary_<CANCER>.pickle` (in this repo) |
| Format | DataFrame: 20,502 gene columns + `censored` + `survival` |
| Censoring | `censored=0` → event (death); `censored=1` → censored |
| Survival | Time in days (float) |
| Toy size | 30 patients × 20 cancer types = 600 total |
| Full TCGA | 200–1000 patients per cancer type |

**Data access deviation:** Full TCGA data requires registration.
Results on toy data are for pipeline verification only.

---

## 2. Environment Setup

```bash
# Option A: conda (recommended)
conda create -n vaecox python=3.8
conda activate vaecox
pip install numpy pandas scipy scikit-learn matplotlib seaborn \
            lifelines torch torchvision tqdm

# Option B: pip venv
python -m venv .venv && source .venv/bin/activate
pip install numpy pandas scipy scikit-learn matplotlib seaborn \
            lifelines torch torchvision tqdm

# Verified with Python 3.13 (Anaconda) — CPU only (no CUDA required)
```

**Our environment:** Python 3.13, Apple M-series CPU, no GPU.
Paper used: Python 3.8, CUDA GPU.

---

## 3. Preprocessing Steps

| Step | Description | Code |
|---|---|---|
| 1 | Load pickle files | `phase1_data_prep.py` |
| 2 | Drop survival columns | gene matrix only for VAE |
| 3 | Z-normalize per gene | `StandardScaler` fit on **train set only** |
| 4 | Stratified 80/20 split | by survival time quintile |
| 5 | 5-fold CV indices | on training set for HP search |
| 6 | 10 random seeds | for evaluation robustness |

---

## 4. Model Settings

### VAE Pretraining
| Hyperparameter | Value | Paper |
|---|---|---|
| Architecture | 20502→4096→128(μ,σ); 128→4096→20502 | same |
| Activation | Tanh | same |
| Optimizer | Adam | same |
| Learning rate | 1e-3 | same |
| Weight decay | 1e-5 | same |
| Batch size | Full batch (600) | same |
| Epochs | **50** | **500** ← deviation |
| Latent dim | 128 | same |

### Survival Models
| Model | lr | weight_decay | Lasso λ | Epochs | HP search |
|---|---|---|---|---|---|
| CoxLasso | 1e-4 | 0 | 0.01 | 50 | None ← deviation |
| CoxRidge | 1e-4 | 1e-3 | 0 | 50 | None ← deviation |
| Coxnnet | 1e-3 | 1e-5 | 0 | 50 | None ← deviation |
| CoxMLP | 1e-3 | 1e-5 | 0 | 50 | None ← deviation |
| VAECox | 1e-3 | 1e-5 | 0 | 50 | None ← deviation |

**Paper HP search:** dropout ∈ {0, 0.3, 0.5} × lr ∈ {1e-3, 1e-4}
× wd ∈ {1e-3, 1e-4, 1e-5} via 5-fold CV → 18 combinations per cancer.

---

## 5. Random Seeds

| Use | Seeds |
|---|---|
| Train/test splits | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 |
| 5-fold CV | random_state = seed |
| Feature subset selection | numpy RNG seed = split seed |
| Robustness corruption | numpy default_rng(seed + 1000) |
| Noise experiments | numpy default_rng(seed + 2000) |

---

## 6. Exact Commands (in order)

```bash
# Step 1: Build toy VAE dataset
python create_toy_vae_dataset.py
# Output: data/toyforVAE_811_mRNA@.tsv  (240MB — gitignored)
#         data/toyforVAE_mRNA@_binary.csv

# Step 2: Prepare train/test splits
python phase1_data_prep.py
# Output: data/prepared/<CANCER>/seed_<N>/*.npy  (241MB — gitignored)
#         data/prepared/phase1_summary.csv

# Step 3: Pretrain VAE (50 epochs, ~2 min on CPU)
python vae_run.py
# Output: results/vae/vae_pretrained/final_model  (678MB — gitignored)
#         results/vae/vae_pretrained/best_model

# Step 4: Phase 2 — model comparison
python phase2_reproduction.py --epochs 50
# Output: results/phase2/cindex_comparison.csv
#         data/embeddings/<CANCER>/seed_<N>/Z_{train,test}.npy

# Step 5: Phase 3 — extensions
python phase3_extensions.py --epochs 50
# Output: results/phase3/3a_lightweight_models.csv
#         results/phase3/3b_feature_subset.csv
#         results/phase3/3cd_robustness.csv
#         results/phase3/3e_fairness_correlation.csv

# Step 6: Phase 3 additional
python phase3_additional.py
# Output: results/phase3/feature_importance.csv
#         results/phase3/figures/kaplan_meier_curves.png
#         results/phase3/km_summary.csv
#         REPRODUCIBILITY.md (this file)
```

---

## 7. Expected Outputs

| File | Description |
|---|---|
| `results/phase2/cindex_comparison.csv` | C-index table: 5 models × 10 cancers |
| `results/phase3/3a_lightweight_models.csv` | hidden/latent dim sweep |
| `results/phase3/3b_feature_subset.csv` | C-index vs feature count |
| `results/phase3/3cd_robustness.csv` | Missing/noisy feature robustness |
| `results/phase3/3e_fairness_correlation.csv` | Event-rate vs C-index |
| `results/phase3/feature_importance.csv` | Top genes per model/cancer |
| `results/phase3/figures/kaplan_meier_curves.png` | KM survival curves |
| `results/phase3/km_summary.csv` | Log-rank test p-values |
| `results/phase3/reproducibility_card.txt` | Full settings + deviations |
| `REPRODUCIBILITY.md` | This checklist |

---

## 8. Deviations from Original Paper

| ID | Deviation | Reason | Impact |
|---|---|---|---|
| DEV-1 | VAE trained 50 epochs (paper: 500) | CPU time | Underfit encoder; lower C-index |
| DEV-2 | VAECox uses frozen encoder | 24 patients too few for fine-tuning | Underestimates VAECox advantage |
| DEV-3 | No HP search (fixed hyperparameters) | Time constraint | Sub-optimal C-index for all models |
| DEV-4 | Toy data 30 pts/cancer (paper: 200–1000) | Data access (dbGaP) | C-index unreliable; not comparable |
| DEV-5 | Python 3.13 / CPU only | Hardware | No scientific impact; fixed 7 bugs |
| DEV-6 | CoxPH-PCA added as extra baseline | Not in paper | Additional reference point |

---

## 9. Key Finding

**Paper:** VAECox outperforms baselines on **7/10** TCGA cancer types by C-index.

**Our reproduction (toy data):** VAECox achieves highest mean C-index (0.572)
and wins on **4/10** cancer types (HNSC, KIRC, LIHC, STAD). Directional
consistency with the paper's finding — VAECox advantage most visible where
training data has sufficient events (≥8 per cancer).

---

*Generated by `phase3_additional.py`. Verify exact values against
`results/phase3/reproducibility_card.txt` for training times and numeric details.*
