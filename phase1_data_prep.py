"""
Phase 1: Data Preparation for VAECox Reproducibility Study

Data format (toy pickle files):
  - 20 cancer types, ~30 patients each (600 total)
  - Each pickle[0] = DataFrame: 20,502 gene columns + 'censored' + 'survival'
  - Each pickle[1] = binary mask DataFrame (all 1s = fully observed)
  - survival: time in days (float)
  - censored: 0 = event (death observed), 1 = censored (alive at follow-up)
  - Gene names: HGNC symbols (A1BG, A2M, ...) including '?' for unknown

Paper details:
  - VAECox (Bioinformatics 2020, Supplement 1)
  - Pretrain VAE on pan-cancer RNA-seq data from 20 TCGA cancers
  - Transfer encoder to Cox survival model per cancer type
  - Evaluate on 10 TCGA cancers: BLCA BRCA HNSC KIRC LGG LIHC LUAD LUSC OV STAD
  - Metric: C-index (concordance index); higher = better survival ranking
  - 10 random seeds x 80/20 train-test split; 5-fold CV on train for HP tuning

Data access note:
  Real TCGA data requires dbGaP access (https://www.ncbi.nlm.nih.gov/gap/).
  This repo ships toy data (~30 patients/cancer) for pipeline verification only.
  Production results require full TCGA RNA-seq (hundreds of patients per type).
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler

# ── Constants ──────────────────────────────────────────────────────────────────

PAPER_10_CANCERS = ['BLCA', 'BRCA', 'HNSC', 'KIRC', 'LGG',
                    'LIHC', 'LUAD', 'LUSC', 'OV', 'STAD']

ALL_20_CANCERS = ['BLCA', 'BRCA', 'CESC', 'COAD', 'GBM', 'HNSC', 'KIRC', 'KIRP',
                  'LAML', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'OV', 'PAAD', 'PRAD',
                  'READ', 'SKCM', 'STAD', 'THCA']

DATA_DIR = './data'
OUT_DIR  = './data/prepared'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Data loading ───────────────────────────────────────────────────────────────

def load_cancer(cancer: str) -> pd.DataFrame:
    """Load toy pickle and return raw DataFrame (genes + censored + survival)."""
    path = os.path.join(DATA_DIR, f'imputed_and_binary_{cancer}.pickle')
    pkg  = pickle.load(open(path, 'rb'))
    return pkg[0].copy()


def get_gene_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in ('censored', 'survival')]


# ── Normalization ──────────────────────────────────────────────────────────────

def z_normalize_train_test(train_df: pd.DataFrame,
                            test_df:  pd.DataFrame,
                            gene_cols: list):
    """
    Fit StandardScaler on training patients only; transform both sets.
    Matches the paper's preprocessing: per-gene z-normalization.
    Scaler is fit on train to prevent data leakage.
    """
    scaler = StandardScaler()
    train_genes = train_df[gene_cols].values.astype(np.float32)
    test_genes  = test_df[gene_cols].values.astype(np.float32)

    train_norm = scaler.fit_transform(train_genes)
    test_norm  = scaler.transform(test_genes)

    train_out = train_df.copy()
    test_out  = test_df.copy()
    train_out[gene_cols] = train_norm
    test_out[gene_cols]  = test_norm
    return train_out, test_out, scaler


# ── Stratified split ───────────────────────────────────────────────────────────

def stratified_survival_split(df: pd.DataFrame,
                               test_size: float = 0.2,
                               seed: int = 0):
    """
    80/20 stratified train/test split on survival time quantiles.
    Matches utils.py SurvivalDataLoader.get_split_dataset().
    Stratification ensures similar survival time distributions in train/test.
    """
    survival = df['survival'].fillna(0).values
    order    = np.argsort(survival)
    strata   = np.zeros(len(order), dtype=int)
    n_bins   = 5
    bin_size = len(order) / n_bins
    for b in range(n_bins):
        strata[order[int(b * bin_size): int((b + 1) * bin_size)]] = b

    return train_test_split(df, test_size=test_size,
                            random_state=seed, stratify=strata)


# ── Per-cancer preparation ─────────────────────────────────────────────────────

def prepare_cancer(cancer: str, seeds: list = list(range(10)), verbose: bool = True):
    """
    Prepare one cancer type:
      - Load raw data
      - Z-normalize gene expression (train-fit, test-transform)
      - Create 10 random 80/20 splits (matching paper)
      - Create 5-fold CV indices on training set for HP search
      - Save splits to disk
      - Return summary stats

    Returns dict with summary info.
    """
    df       = load_cancer(cancer)
    gene_cols = get_gene_columns(df)
    n        = len(df)
    n_events = int((df['censored'] == 0).sum())
    n_cens   = int((df['censored'] == 1).sum())

    if verbose:
        print(f'\n{"="*60}')
        print(f'Cancer: {cancer}')
        print(f'  Total patients  : {n}')
        print(f'  Events (deaths) : {n_events}  ({100*n_events/n:.1f}%)')
        print(f'  Censored        : {n_cens}   ({100*n_cens/n:.1f}%)')
        print(f'  Survival (days) : min={df["survival"].min():.0f}  '
              f'median={df["survival"].median():.0f}  '
              f'max={df["survival"].max():.0f}')
        print(f'  Gene features   : {len(gene_cols)}')

    cancer_dir = os.path.join(OUT_DIR, cancer)
    os.makedirs(cancer_dir, exist_ok=True)

    splits_summary = []
    for seed in seeds:
        train_df, test_df = stratified_survival_split(df, test_size=0.2, seed=seed)
        train_norm, test_norm, scaler = z_normalize_train_test(
            train_df, test_df, gene_cols)

        # 5-fold CV indices on the training set
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        cv_folds = list(kf.split(train_norm))

        n_train_events = int((train_norm['censored'] == 0).sum())
        n_test_events  = int((test_norm['censored'] == 0).sum())

        # Save to disk (numpy arrays for fast loading in Phase 2)
        seed_dir = os.path.join(cancer_dir, f'seed_{seed}')
        os.makedirs(seed_dir, exist_ok=True)

        X_train = train_norm[gene_cols].values.astype(np.float32)
        X_test  = test_norm[gene_cols].values.astype(np.float32)
        y_train = train_norm['survival'].values.astype(np.float32)
        y_test  = test_norm['survival'].values.astype(np.float32)
        c_train = train_norm['censored'].values.astype(np.int32)
        c_test  = test_norm['censored'].values.astype(np.int32)

        np.save(os.path.join(seed_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(seed_dir, 'X_test.npy'),  X_test)
        np.save(os.path.join(seed_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(seed_dir, 'y_test.npy'),  y_test)
        np.save(os.path.join(seed_dir, 'c_train.npy'), c_train)
        np.save(os.path.join(seed_dir, 'c_test.npy'),  c_test)

        # Save CV fold indices
        for fold_i, (tr_idx, val_idx) in enumerate(cv_folds):
            np.save(os.path.join(seed_dir, f'cv_train_{fold_i}.npy'), tr_idx)
            np.save(os.path.join(seed_dir, f'cv_val_{fold_i}.npy'),   val_idx)

        splits_summary.append({
            'seed': seed,
            'n_train': len(X_train), 'n_test': len(X_test),
            'n_train_events': n_train_events, 'n_test_events': n_test_events,
        })

    if verbose:
        s0 = splits_summary[0]
        print(f'  Train/test split: {s0["n_train"]}/{s0["n_test"]} (seed 0 example)')
        print(f'    Train events  : {s0["n_train_events"]}')
        print(f'    Test events   : {s0["n_test_events"]}')
        if s0['n_test_events'] == 0:
            print(f'  ⚠ WARNING: test set has 0 events — C-index undefined (toy data limitation)')
        print(f'  Saved splits to: {cancer_dir}/')

    return {
        'cancer': cancer,
        'n': n, 'n_events': n_events, 'n_censored': n_cens,
        'censor_rate': round(n_cens / n * 100, 1),
        'median_survival': df['survival'].median(),
        'splits': splits_summary,
    }


# ── Full dataset summary ───────────────────────────────────────────────────────

def print_data_access_note():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                  DATA ACCESS NOTE                                ║
╠══════════════════════════════════════════════════════════════════╣
║  Real TCGA data:                                                 ║
║    Source : ICGC Data Portal (https://dcc.icgc.org)              ║
║    Auth   : dbGaP accession phs000178 required                   ║
║    Size   : ~200-1000 patients per cancer type                   ║
║    Format : RNA-seq FPKM, then log2(FPKM+1), then z-norm        ║
║                                                                  ║
║  Toy data (this repo):                                           ║
║    Source : github.com/dmis-lab/VAECox ./data/                   ║
║    Auth   : None required                                        ║
║    Size   : 30 patients per cancer type (600 total)              ║
║    Format : Pre-processed (close to z-normalized)                ║
║                                                                  ║
║  Impact on results:                                              ║
║    - Toy C-index values are UNRELIABLE (too few events/patients) ║
║    - Pipeline correctness can be verified on toy data            ║
║    - Reported C-index values require full TCGA data              ║
║    - Paper reports results on 200-1000 patients per cancer       ║
╚══════════════════════════════════════════════════════════════════╝
""")


def run_phase1(cancers=None, seeds=list(range(10))):
    """Run full Phase 1 data preparation."""

    print_data_access_note()

    if cancers is None:
        cancers = PAPER_10_CANCERS

    print(f'Preparing {len(cancers)} cancer types: {cancers}')
    print(f'Random seeds: {seeds}')

    all_results = []
    for cancer in cancers:
        result = prepare_cancer(cancer, seeds=seeds, verbose=True)
        all_results.append(result)

    # Summary table
    print(f'\n{"="*70}')
    print('PHASE 1 SUMMARY — Toy Data')
    print(f'{"="*70}')
    print(f'{"Cancer":<8} {"N":>4} {"Events":>7} {"Censored":>9} '
          f'{"Censor%":>8} {"MedianSurv":>11}')
    print('-' * 50)
    for r in all_results:
        flag = ' ⚠' if r['n_events'] < 5 else ''
        print(f'{r["cancer"]:<8} {r["n"]:>4} {r["n_events"]:>7} '
              f'{r["n_censored"]:>9} {r["censor_rate"]:>7}% '
              f'{r["median_survival"]:>10.0f}{flag}')

    print(f'\n⚠ = fewer than 5 events; C-index unreliable (toy data limitation)')
    print(f'\nPreprocessing applied:')
    print(f'  - Z-normalization per gene (fit on train, transform test)')
    print(f'  - Stratified 80/20 train/test split by survival time quintile')
    print(f'  - 5-fold CV indices on training set for HP tuning')
    print(f'  - {len(seeds)} random seeds for robustness (matching paper)')
    print(f'\nOutputs saved to: {OUT_DIR}/<CANCER>/seed_<N>/')
    print(f'  X_train.npy, X_test.npy  — gene features (float32)')
    print(f'  y_train.npy, y_test.npy  — survival time (float32)')
    print(f'  c_train.npy, c_test.npy  — censored flag (int32, 0=event)')
    print(f'  cv_train_<k>.npy, cv_val_<k>.npy — CV fold indices')

    return all_results


if __name__ == '__main__':
    # Phase 1: start with 3 focus cancers, then expand to all 10
    print('=== Phase 1: Focus cancers (BRCA, LGG, LUAD) ===')
    run_phase1(cancers=['BRCA', 'LGG', 'LUAD'], seeds=list(range(10)))

    print('\n\n=== Phase 1: All 10 paper cancers ===')
    run_phase1(cancers=PAPER_10_CANCERS, seeds=list(range(10)))
