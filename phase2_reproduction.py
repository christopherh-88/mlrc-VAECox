"""
Phase 2: VAECox Reproduction — Model Comparison

Models:
  CoxLasso   — linear Cox with L1 (LASSO) penalty on fc1 weights
  CoxRidge   — linear Cox with L2 (Ridge) penalty via weight_decay
  Coxnnet    — 2-layer neural Cox (sqrt(p) hidden units)
  VAECox     — pretrained VAE encoder (frozen, 128-d) + Coxnnet head
                Note: we freeze the encoder for speed (24 patients vs 4096+128
                parameters makes end-to-end fine-tuning prone to collapse).
                The paper fine-tunes fully on 200-1000 patients/cancer.
  CoxPH-PCA  — lifelines Cox PH on top-10 PCA components (extra baseline)

Evaluation:
  10 random seeds, 80/20 stratified train/test split (Phase 1 splits)
  C-index on held-out test set; mean ± std across valid seeds
  (seeds where test set has ≥1 event; BRCA/LUAD/PRAD often have 0)

Toy data limitation:
  30 patients/cancer; paper uses 200–1000. C-index values are noisy and
  NOT directly comparable to the paper's Table 1. The goal here is to
  verify the pipeline end-to-end and confirm VAECox embeddings are useful.
"""

import os, sys, types, logging, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lifelines.utils import concordance_index
from lifelines import CoxPHFitter
from sklearn.decomposition import PCA
from tqdm import tqdm

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
from models import CoxRegression, Coxnnet, PartialNLL
import vae_models as vae_mod

# ── Constants ──────────────────────────────────────────────────────────────────
PAPER_10 = ['BLCA', 'BRCA', 'HNSC', 'KIRC', 'LGG',
            'LIHC', 'LUAD', 'LUSC', 'OV', 'STAD']

PRETRAINED_VAE = 'results/vae/vae_pretrained/final_model'
DATA_DIR       = 'data/prepared'
EMBED_DIR      = 'data/embeddings'
RESULTS_DIR    = 'results/phase2'
for d in [EMBED_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── VAE config namespace ───────────────────────────────────────────────────────

def _vae_config():
    c = types.SimpleNamespace()
    c.hidden_nodes   = 4096
    c.acti_func      = 'Tanh'
    c.dropout_rate   = 0.0
    c.model_type     = 'vae'
    c.session_name   = 'vae_pretrained'
    c.max_epochs     = 500
    c.learning_rate  = 1e-3
    c.model_optimizer = 'Adam'
    c.weight_sparsity = 1e-6
    c.weight_decay   = 1e-5
    c.save_mode      = False
    c.device_type    = 'cpu'
    c.exclude_impute = False
    c.batch_size     = 0
    c.vae_data       = 'toyforVAE'
    c.model_struct   = 'basic'
    return c

# ── Pre-compute VAE embeddings (done once, cached to disk) ─────────────────────

def load_vae_encoder(config):
    """Load pretrained VAE encoder weights (once into memory)."""
    LOGGER = logging.getLogger('vae')
    vae = vae_mod.VAE(config, LOGGER, num_features=20502)
    ckpt = torch.load(PRETRAINED_VAE, map_location='cpu', weights_only=False)
    vae.load_state_dict(ckpt['model_state_dict'])
    vae = vae.float().eval()
    print(f'Loaded pretrained VAE: {PRETRAINED_VAE}')
    print(f'  Encoder: 20502 → 4096 → 128 (Tanh activations)')
    return vae


def compute_embeddings(vae, X: np.ndarray) -> np.ndarray:
    """Forward pass through frozen encoder; return 128-d mu."""
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32)
        h   = vae.encode(X_t)
        mu  = vae.encode_mu(h)
    return mu.numpy()


def precompute_all_embeddings(cancers=PAPER_10, seeds=range(10)):
    """
    For each (cancer, seed), compute train+test VAE embeddings and cache
    to EMBED_DIR. Skips if already cached.
    """
    config = _vae_config()
    vae    = load_vae_encoder(config)

    print('\nPre-computing VAE embeddings (cached to disk)...')
    for cancer in tqdm(cancers, desc='Cancers'):
        for seed in seeds:
            emb_dir = f'{EMBED_DIR}/{cancer}/seed_{seed}'
            train_path = f'{emb_dir}/Z_train.npy'
            test_path  = f'{emb_dir}/Z_test.npy'
            if os.path.exists(train_path) and os.path.exists(test_path):
                continue
            os.makedirs(emb_dir, exist_ok=True)
            X_tr = np.load(f'{DATA_DIR}/{cancer}/seed_{seed}/X_train.npy')
            X_te = np.load(f'{DATA_DIR}/{cancer}/seed_{seed}/X_test.npy')
            Z_tr = compute_embeddings(vae, X_tr)
            Z_te = compute_embeddings(vae, X_te)
            np.save(train_path, Z_tr)
            np.save(test_path,  Z_te)
    print('Embeddings ready.')

# ── Data loading ───────────────────────────────────────────────────────────────

def load_split(cancer: str, seed: int):
    d = f'{DATA_DIR}/{cancer}/seed_{seed}'
    return (
        np.load(f'{d}/X_train.npy').astype(np.float32),
        np.load(f'{d}/X_test.npy').astype(np.float32),
        np.load(f'{d}/y_train.npy').astype(np.float64),
        np.load(f'{d}/y_test.npy').astype(np.float64),
        np.load(f'{d}/c_train.npy').astype(np.int32),
        np.load(f'{d}/c_test.npy').astype(np.int32),
    )


def load_embeddings(cancer: str, seed: int):
    d = f'{EMBED_DIR}/{cancer}/seed_{seed}'
    return (
        np.load(f'{d}/Z_train.npy').astype(np.float32),
        np.load(f'{d}/Z_test.npy').astype(np.float32),
    )

# ── Risk set matrix ────────────────────────────────────────────────────────────

def make_R(y: np.ndarray) -> np.ndarray:
    n = len(y)
    R = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        R[i, :] = (y >= y[i])
    return R

# ── C-index wrapper ────────────────────────────────────────────────────────────

def cindex_safe(y, pred, c) -> float:
    event_obs = (c == 0)
    if event_obs.sum() == 0:
        return float('nan')
    try:
        return concordance_index(y, pred, event_obs)
    except Exception:
        return float('nan')

# ── Neural Cox training ────────────────────────────────────────────────────────

def train_and_eval(model, X_tr, y_tr, c_tr, X_te, y_te, c_te,
                   lr=1e-3, wd=1e-5, epochs=50, lasso_lambda=0.0) -> float:
    """Full-batch Cox partial-NLL training + test C-index."""
    model = model.float()
    loss_fn = PartialNLL()

    X  = torch.tensor(X_tr, dtype=torch.float32)
    c  = torch.tensor(c_tr, dtype=torch.float32)
    R  = torch.tensor(make_R(y_tr), dtype=torch.float32)

    # PartialNLL expects double; cast inside the loss
    model = model.double()
    X  = X.double()
    c  = c.double()
    R  = R.double()

    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    model.train()

    for _ in range(epochs):
        optim.zero_grad()
        theta = model(X)
        if isinstance(theta, tuple):
            theta = theta[0]
        loss = loss_fn(theta, R, c)
        if lasso_lambda > 0:
            L1 = nn.L1Loss()
            params = torch.cat([p.view(-1) for p in model.fc1.parameters()])
            loss = loss + lasso_lambda * L1(params, torch.zeros_like(params))
        if torch.isnan(loss) or torch.isinf(loss):
            break
        loss.backward()
        optim.step()

    model.eval()
    with torch.no_grad():
        Xte = torch.tensor(X_te, dtype=torch.float64)
        theta_te = model(Xte)
        if isinstance(theta_te, tuple):
            theta_te = theta_te[0]
        pred = -theta_te.reshape(-1).numpy()

    return cindex_safe(y_te, pred, c_te)

# ── Per-model runners ──────────────────────────────────────────────────────────

def run_coxlasso(X_tr, y_tr, c_tr, X_te, y_te, c_te, epochs=50) -> float:
    model = CoxRegression(X_tr.shape[1])
    return train_and_eval(model, X_tr, y_tr, c_tr, X_te, y_te, c_te,
                          lr=1e-4, wd=0, epochs=epochs, lasso_lambda=0.01)


def run_coxridge(X_tr, y_tr, c_tr, X_te, y_te, c_te, epochs=50) -> float:
    model = CoxRegression(X_tr.shape[1])
    return train_and_eval(model, X_tr, y_tr, c_tr, X_te, y_te, c_te,
                          lr=1e-4, wd=1e-3, epochs=epochs)


def run_coxnnet(X_tr, y_tr, c_tr, X_te, y_te, c_te, epochs=50) -> float:
    model = Coxnnet(X_tr.shape[1])
    return train_and_eval(model, X_tr, y_tr, c_tr, X_te, y_te, c_te,
                          lr=1e-3, wd=1e-5, epochs=epochs)


def run_vaecox(Z_tr, y_tr, c_tr, Z_te, y_te, c_te, epochs=50) -> float:
    """Coxnnet on frozen 128-d VAE embeddings."""
    model = Coxnnet(Z_tr.shape[1])   # Coxnnet(128)
    return train_and_eval(model, Z_tr, y_tr, c_tr, Z_te, y_te, c_te,
                          lr=1e-3, wd=1e-5, epochs=epochs)


def run_coxph_pca(X_tr, y_tr, c_tr, X_te, y_te, c_te,
                  n_components=10, penalizer=1.0) -> float:
    try:
        n_comp = min(n_components, X_tr.shape[0] - 2)
        if n_comp < 1:
            return float('nan')
        pca = PCA(n_components=n_comp, random_state=42)
        Ztr = pca.fit_transform(X_tr)
        Zte = pca.transform(X_te)
        cols = [f'PC{i}' for i in range(n_comp)]
        df_tr = pd.DataFrame(Ztr, columns=cols)
        df_tr['T'] = y_tr
        df_tr['E'] = (c_tr == 0).astype(int)
        cph = CoxPHFitter(penalizer=penalizer, l1_ratio=0.5)
        cph.fit(df_tr, duration_col='T', event_col='E', show_progress=False)
        df_te = pd.DataFrame(Zte, columns=cols)
        pred  = cph.predict_log_partial_hazard(df_te).values
        return cindex_safe(y_te, pred, c_te)
    except Exception:
        return float('nan')

# ── Main evaluation ────────────────────────────────────────────────────────────

MODEL_NAMES = ['CoxLasso', 'CoxRidge', 'Coxnnet', 'VAECox', 'CoxPH-PCA']


def run_phase2(cancers=PAPER_10, seeds=range(10), epochs=50, verbose=True):
    results = {m: {c: [] for c in cancers} for m in MODEL_NAMES}

    for cancer in cancers:
        if verbose:
            print(f'\n── {cancer} ──────────────────────────────────────────')
        for seed in tqdm(list(seeds), desc=cancer, leave=False, disable=not verbose):
            X_tr, X_te, y_tr, y_te, c_tr, c_te = load_split(cancer, seed)
            Z_tr, Z_te                           = load_embeddings(cancer, seed)

            results['CoxLasso'][cancer].append(
                run_coxlasso(X_tr, y_tr, c_tr, X_te, y_te, c_te, epochs))
            results['CoxRidge'][cancer].append(
                run_coxridge(X_tr, y_tr, c_tr, X_te, y_te, c_te, epochs))
            results['Coxnnet'][cancer].append(
                run_coxnnet(X_tr, y_tr, c_tr, X_te, y_te, c_te, epochs))
            results['VAECox'][cancer].append(
                run_vaecox(Z_tr, y_tr, c_tr, Z_te, y_te, c_te, epochs))
            results['CoxPH-PCA'][cancer].append(
                run_coxph_pca(X_tr, y_tr, c_tr, X_te, y_te, c_te))

        if verbose:
            for mname in MODEL_NAMES:
                vals = [v for v in results[mname][cancer] if not np.isnan(v)]
                if vals:
                    print(f'  {mname:<12}: {np.mean(vals):.3f} ± {np.std(vals):.3f}'
                          f'  ({len(vals)}/{len(list(seeds))} valid seeds)')
                else:
                    print(f'  {mname:<12}: ⊘  (0 test events across all seeds)')

    return results

# ── Results reporting ──────────────────────────────────────────────────────────

def make_results_table(results, cancers):
    rows = []
    for mname in MODEL_NAMES:
        row = {'Model': mname}
        all_means = []
        for cancer in cancers:
            vals = [v for v in results[mname][cancer] if not np.isnan(v)]
            if vals:
                mu = np.mean(vals)
                sd = np.std(vals)
                row[cancer] = f'{mu:.3f}'
                all_means.append(mu)
            else:
                row[cancer] = ' ⊘ '
        row['Mean'] = f'{np.mean(all_means):.3f}' if all_means else '⊘'
        rows.append(row)
    return pd.DataFrame(rows).set_index('Model')


def count_wins(results, cancers):
    wins = {m: 0 for m in MODEL_NAMES}
    for cancer in cancers:
        best_ci, best_m = -1, None
        for mname in MODEL_NAMES:
            vals = [v for v in results[mname][cancer] if not np.isnan(v)]
            if vals and np.mean(vals) > best_ci:
                best_ci = np.mean(vals)
                best_m  = mname
        if best_m:
            wins[best_m] += 1
    return wins


def print_summary(results, cancers):
    table = make_results_table(results, cancers)
    wins  = count_wins(results, cancers)

    print('\n' + '=' * 80)
    print('PHASE 2 RESULTS — C-index (mean across seeds, ⊘ = no test events)')
    print('=' * 80)
    print(table.to_string())
    print()
    print('Best model per cancer (most wins):')
    for m, w in sorted(wins.items(), key=lambda x: -x[1]):
        bar = '█' * w
        print(f'  {m:<12} {bar:<12} {w}/{len(cancers)}')
    print()
    print('Reference (paper, full TCGA data):')
    paper_wins = {
        'CoxLasso': 0, 'CoxRidge': 0, 'Coxnnet': 2, 'VAECox': 7, 'CoxPH-PCA': 0
    }
    for m, w in sorted(paper_wins.items(), key=lambda x: -x[1]):
        bar = '█' * w
        print(f'  {m:<12} {bar:<12} {w}/10  (reported)')
    print()
    print('Note: toy-data results are not comparable to the paper.')
    print('      VAECox frozen-encoder may differ from paper\'s fine-tuned variant.')

    # Save numeric CSV
    numeric = {}
    for mname in MODEL_NAMES:
        row = {}
        for cancer in cancers:
            vals = [v for v in results[mname][cancer] if not np.isnan(v)]
            row[cancer] = round(np.mean(vals), 4) if vals else float('nan')
        row['Mean'] = round(np.nanmean(list(row.values())), 4)
        numeric[mname] = row
    df_num = pd.DataFrame(numeric).T
    df_num.index.name = 'Model'
    out_path = f'{RESULTS_DIR}/cindex_comparison.csv'
    df_num.to_csv(out_path)
    print(f'\nSaved: {out_path}')
    return table


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--cancers', nargs='+', default=PAPER_10)
    p.add_argument('--seeds',   nargs='+', type=int, default=list(range(10)))
    p.add_argument('--epochs',  type=int, default=50)
    args = p.parse_args()

    print('Phase 2: VAECox Reproduction')
    print(f'  Cancers : {args.cancers}')
    print(f'  Seeds   : {args.seeds}')
    print(f'  Epochs  : {args.epochs}')

    # Step 1: Pre-compute and cache VAE embeddings (run once)
    precompute_all_embeddings(cancers=args.cancers, seeds=args.seeds)

    # Step 2: Train all models and collect C-index
    results = run_phase2(cancers=args.cancers, seeds=args.seeds,
                         epochs=args.epochs)

    # Step 3: Print and save results
    print_summary(results, args.cancers)
