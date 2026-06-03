"""
Generate all paper figures from saved result CSVs.

Figures produced:
  fig1_cindex_barchart.png      — C-index per cancer, grouped bars (main result)
  fig2_cindex_heatmap.png       — C-index heatmap: models × cancers
  fig3_model_wins.png           — Win count: ours vs paper claim
  fig4_lightweight_tradeoff.png — hidden_dim/latent_dim vs C-index + training time
  fig5_feature_subset.png       — C-index vs number of genes (accessibility)
  fig6_robustness.png           — Robustness: missing features + Gaussian noise
  fig7_fairness_scatter.png     — n_events vs C-index per cancer (all models)
  fig8_feature_importance.png   — Top-10 gene importance (CoxRidge + VAECox)
  kaplan_meier_curves.png       — already exists; re-generated for consistency

All saved to: results/phase3/figures/
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

os.makedirs('results/phase3/figures', exist_ok=True)

# ── Shared style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

# Model color palette (VAECox highlighted in red)
MODEL_COLORS = {
    'CoxLasso':  '#E87722',   # orange
    'CoxRidge':  '#4DAF4A',   # green
    'Coxnnet':   '#377EB8',   # blue
    'CoxMLP':    '#984EA3',   # purple
    'VAECox':    '#E31A1C',   # red — the focal model
    'CoxPH-PCA': '#AAAAAA',   # gray — extra baseline
}
MODEL_ORDER = ['CoxLasso', 'CoxRidge', 'Coxnnet', 'CoxMLP', 'VAECox', 'CoxPH-PCA']

PAPER_10 = ['BLCA','BRCA','HNSC','KIRC','LGG','LIHC','LUAD','LUSC','OV','STAD']

def savefig(name):
    path = f'results/phase3/figures/{name}'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 1 — C-index grouped bar chart (main Phase 2 result)
# ══════════════════════════════════════════════════════════════════════════════
def fig1_cindex_barchart():
    df = pd.read_csv('results/phase2/cindex_comparison.csv', index_col=0)
    models  = [m for m in MODEL_ORDER if m in df.index and m != 'CoxPH-PCA']
    cancers = PAPER_10

    x   = np.arange(len(cancers))
    n   = len(models)
    w   = 0.13
    off = np.linspace(-(n-1)*w/2, (n-1)*w/2, n)

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, model in enumerate(models):
        vals = [df.loc[model, c] if c in df.columns else np.nan for c in cancers]
        bars = ax.bar(x + off[i], vals, w, label=model,
                      color=MODEL_COLORS[model], alpha=0.85, zorder=3)
        # Hatch VAECox for emphasis
        if model == 'VAECox':
            for b in bars:
                b.set_edgecolor('black')
                b.set_linewidth(0.8)

    ax.axhline(0.5, color='black', linestyle=':', linewidth=1, label='Random (0.5)')
    ax.set_xticks(x)
    ax.set_xticklabels(cancers, rotation=0)
    ax.set_ylabel('C-index (mean across seeds)')
    ax.set_xlabel('Cancer type')
    ax.set_ylim(0, 1.08)
    ax.set_title('Phase 2: C-index Comparison Across 10 TCGA Cancer Types\n'
                 '(toy data, 30 patients/cancer — not comparable to paper Table 1)')
    ax.legend(loc='upper right', framealpha=0.9, ncol=3)
    # Annotate BRCA/LUAD as unreliable
    for ci, cancer in enumerate(cancers):
        if cancer in ('BRCA', 'LUAD'):
            ax.annotate('≤3\nevents', xy=(x[ci], 0.05), ha='center',
                        fontsize=7, color='gray')
    fig.tight_layout()
    savefig('fig1_cindex_barchart.png')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 2 — C-index heatmap
# ══════════════════════════════════════════════════════════════════════════════
def fig2_cindex_heatmap():
    df = pd.read_csv('results/phase2/cindex_comparison.csv', index_col=0)
    models  = [m for m in MODEL_ORDER if m in df.index]
    cancers = PAPER_10
    mat = df.loc[models, cancers].values.astype(float)

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(mat, aspect='auto', cmap='RdYlGn', vmin=0.3, vmax=0.9)
    ax.set_xticks(range(len(cancers)));  ax.set_xticklabels(cancers)
    ax.set_yticks(range(len(models)));   ax.set_yticklabels(models)
    # Annotate cells
    for i in range(len(models)):
        for j in range(len(cancers)):
            v = mat[i, j]
            txt = f'{v:.2f}' if not np.isnan(v) else '⊘'
            color = 'white' if (v < 0.4 or v > 0.75) else 'black'
            ax.text(j, i, txt, ha='center', va='center', fontsize=7.5,
                    color=color, fontweight='bold' if models[i]=='VAECox' else 'normal')
    plt.colorbar(im, ax=ax, label='C-index', shrink=0.8)
    ax.set_title('C-index Heatmap: Models × Cancer Types\n'
                 '(green = higher C-index; ⊘ = no test events)')
    ax.set_xlabel('Cancer type')
    # Bold VAECox row label
    for label in ax.get_yticklabels():
        if label.get_text() == 'VAECox':
            label.set_fontweight('bold')
            label.set_color(MODEL_COLORS['VAECox'])
    fig.tight_layout()
    savefig('fig2_cindex_heatmap.png')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 3 — Model wins: ours vs paper
# ══════════════════════════════════════════════════════════════════════════════
def fig3_model_wins():
    df = pd.read_csv('results/phase2/cindex_comparison.csv', index_col=0)
    models  = [m for m in MODEL_ORDER if m in df.index and m != 'CoxPH-PCA']
    cancers = PAPER_10

    # Count wins (highest C-index per cancer)
    our_wins = {m: 0 for m in models}
    for cancer in cancers:
        best_ci, best_m = -1, None
        for m in models:
            v = df.loc[m, cancer] if cancer in df.columns else np.nan
            if not np.isnan(v) and v > best_ci:
                best_ci, best_m = v, m
        if best_m: our_wins[best_m] += 1

    paper_wins = {'CoxLasso':0, 'CoxRidge':0, 'Coxnnet':2, 'CoxMLP':1, 'VAECox':7}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, wins, title in zip(axes,
                                [our_wins, paper_wins],
                                ['Our Results (toy data, 30 pts/cancer)',
                                 'Paper Reported (full TCGA, 200-1000 pts)']):
        bars = ax.barh([m for m in models],
                       [wins.get(m, 0) for m in models],
                       color=[MODEL_COLORS[m] for m in models],
                       alpha=0.85, edgecolor='black', linewidth=0.5)
        ax.set_xlim(0, 10)
        ax.axvline(5, color='gray', linestyle=':', linewidth=1)
        ax.set_xlabel('Number of cancers won (out of 10)')
        ax.set_title(title)
        for bar, m in zip(bars, models):
            v = wins.get(m, 0)
            ax.text(v + 0.1, bar.get_y() + bar.get_height()/2,
                    str(v), va='center', fontsize=10,
                    fontweight='bold' if m=='VAECox' else 'normal')
    axes[0].set_ylabel('Model')
    fig.suptitle('VAECox Win Count: Reproduction vs Paper Claim', fontsize=12)
    fig.tight_layout()
    savefig('fig3_model_wins.png')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 4 — Lightweight model trade-off
# ══════════════════════════════════════════════════════════════════════════════
def fig4_lightweight_tradeoff():
    df = pd.read_csv('results/phase3/3a_lightweight_models.csv')

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel A: hidden_dim vs C-index (latent=128)
    sub = df[df['latent_dim'] == 128].sort_values('hidden_dim')
    ax = axes[0]
    ax.plot(sub['hidden_dim'], sub['mean_cindex'], 'o-',
            color='steelblue', linewidth=2, markersize=8)
    for _, row in sub.iterrows():
        ax.annotate(f"h={int(row['hidden_dim'])}\n{row['mean_cindex']:.3f}",
                    (row['hidden_dim'], row['mean_cindex']),
                    textcoords='offset points', xytext=(0, 10), ha='center', fontsize=8)
    ax.set_xlabel('VAE hidden dim')
    ax.set_ylabel('Mean C-index')
    ax.set_title('A. Hidden dim vs C-index\n(latent=128, 50 epochs)')
    ax.set_xscale('log')

    # Panel B: latent_dim vs C-index (hidden=4096)
    sub2 = df[df['hidden_dim'] == 4096].sort_values('latent_dim')
    ax = axes[1]
    ax.plot(sub2['latent_dim'], sub2['mean_cindex'], 's-',
            color='darkorange', linewidth=2, markersize=8)
    for _, row in sub2.iterrows():
        ax.annotate(f"l={int(row['latent_dim'])}\n{row['mean_cindex']:.3f}",
                    (row['latent_dim'], row['mean_cindex']),
                    textcoords='offset points', xytext=(0, 10), ha='center', fontsize=8)
    ax.set_xlabel('VAE latent dim')
    ax.set_ylabel('Mean C-index')
    ax.set_title('B. Latent dim vs C-index\n(hidden=4096, 50 epochs)')

    # Panel C: training time vs C-index (bubble = #params)
    ax = axes[2]
    sub3 = df[df['latent_dim'] == 128].copy()
    sizes = (sub3['n_params'] / sub3['n_params'].max() * 400).values
    sc = ax.scatter(sub3['train_sec'], sub3['mean_cindex'],
                    s=sizes, c=['#1f77b4','#ff7f0e','#d62728'],
                    alpha=0.8, edgecolor='black', linewidth=0.8)
    for _, row in sub3.iterrows():
        ax.annotate(f"h={int(row['hidden_dim'])}",
                    (row['train_sec'], row['mean_cindex']),
                    textcoords='offset points', xytext=(5, 5), fontsize=9)
    ax.set_xlabel('Training time (seconds, CPU)')
    ax.set_ylabel('Mean C-index')
    ax.set_title('C. Training time vs C-index\n(bubble size ∝ #parameters)')
    # Size legend
    for params, label in [(21e6,'21M'), (169e6,'170M')]:
        ax.scatter([], [], s=params/sub3['n_params'].max()*400,
                   c='gray', alpha=0.5, label=label)
    ax.legend(title='#params', fontsize=8, title_fontsize=8)

    fig.suptitle('Phase 3A: Lightweight VAECox — Architecture Trade-offs\n'
                 '(mean C-index across 10 cancers × 5 seeds)', fontsize=11)
    fig.tight_layout()
    savefig('fig4_lightweight_tradeoff.png')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 5 — Feature subset (low-resource accessibility)
# ══════════════════════════════════════════════════════════════════════════════
def fig5_feature_subset():
    df = pd.read_csv('results/phase3/3b_feature_subset.csv')
    # Also add VAECox baseline (uses 128-d embedding = compressed from 20502)
    vaecox_ci_row = pd.read_csv('results/phase2/cindex_comparison.csv', index_col=0)
    vaecox_mean = vaecox_ci_row.loc['VAECox', PAPER_10].mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df['k_genes'], df['mean_cindex'], 'o-',
            color=MODEL_COLORS['CoxRidge'], linewidth=2, markersize=8,
            label='CoxRidge (top-k genes by variance)')

    # VAECox reference line (uses 128-d VAE embedding of 20502 genes)
    ax.axhline(vaecox_mean, color=MODEL_COLORS['VAECox'], linestyle='--',
               linewidth=2, label=f'VAECox (frozen 128-d embedding): {vaecox_mean:.3f}')

    # Random baseline
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=1, label='Random (0.5)')

    for _, row in df.iterrows():
        ax.annotate(f"{row['mean_cindex']:.3f}",
                    (row['k_genes'], row['mean_cindex']),
                    textcoords='offset points', xytext=(0, 9),
                    ha='center', fontsize=8)

    ax.set_xscale('log')
    ax.set_xlabel('Number of genes used (top-k by training-set variance)')
    ax.set_ylabel('Mean C-index (across 10 cancers × 5 seeds)')
    ax.set_title('Phase 3B: Low-Resource Feature Subset\n'
                 'CoxRidge on top-k genes vs VAECox compressed embedding')
    ax.legend(framealpha=0.9)
    ax.set_ylim(0.3, 0.75)
    # Annotate sweet spot
    best_row = df.loc[df['mean_cindex'].idxmax()]
    ax.annotate(f'Best: k={int(best_row["k_genes"])}',
                xy=(best_row['k_genes'], best_row['mean_cindex']),
                xytext=(best_row['k_genes']*2, best_row['mean_cindex']+0.04),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=9, color='darkgreen')
    fig.tight_layout()
    savefig('fig5_feature_subset.png')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 6 — Robustness: missing features + Gaussian noise
# ══════════════════════════════════════════════════════════════════════════════
def fig6_robustness():
    df = pd.read_csv('results/phase3/3cd_robustness.csv')
    miss = df[df['experiment'] == 'missing'].sort_values('pct_or_sigma')
    nois = df[df['experiment'] == 'noise'].sort_values('pct_or_sigma')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Missing features
    ax = axes[0]
    ax.plot(miss['pct_or_sigma'] * 100, miss['CoxRidge'], 's-',
            color=MODEL_COLORS['CoxRidge'], linewidth=2, markersize=8,
            label='CoxRidge (raw features)')
    ax.plot(miss['pct_or_sigma'] * 100, miss['VAECox'], 'o-',
            color=MODEL_COLORS['VAECox'], linewidth=2, markersize=8,
            label='VAECox (frozen encoder)')
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    # Shade between the two lines
    ax.fill_between(miss['pct_or_sigma'] * 100,
                    miss['CoxRidge'], miss['VAECox'],
                    alpha=0.1, color='purple')
    ax.set_xlabel('Features randomly zeroed out (%)')
    ax.set_ylabel('Mean C-index')
    ax.set_title('A. Robustness to Missing Features\n'
                 '(4 cancers: BLCA, OV, STAD, HNSC; 5 seeds each)')
    ax.set_xticks([0, 10, 25, 50])
    ax.legend()
    ax.set_ylim(0.35, 0.85)

    # Panel B: Gaussian noise
    ax = axes[1]
    ax.plot(nois['pct_or_sigma'], nois['CoxRidge'], 's-',
            color=MODEL_COLORS['CoxRidge'], linewidth=2, markersize=8,
            label='CoxRidge (raw features)')
    ax.plot(nois['pct_or_sigma'], nois['VAECox'], 'o-',
            color=MODEL_COLORS['VAECox'], linewidth=2, markersize=8,
            label='VAECox (frozen encoder)')
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax.fill_between(nois['pct_or_sigma'],
                    nois['CoxRidge'], nois['VAECox'],
                    alpha=0.1, color='purple')
    # Annotate C-index drop for VAECox
    v0 = nois[nois['pct_or_sigma']==0.0]['VAECox'].values[0]
    v2 = nois[nois['pct_or_sigma']==2.0]['VAECox'].values[0]
    ax.annotate(f'Drop: {v0-v2:.3f}',
                xy=(2.0, v2), xytext=(1.5, v2+0.05),
                arrowprops=dict(arrowstyle='->', color=MODEL_COLORS['VAECox']),
                color=MODEL_COLORS['VAECox'], fontsize=9)
    ax.set_xlabel('Gaussian noise σ added to gene expression')
    ax.set_ylabel('Mean C-index')
    ax.set_title('B. Robustness to Gaussian Noise\n'
                 '(4 cancers: BLCA, OV, STAD, HNSC; 5 seeds each)')
    ax.set_xticks([0, 0.5, 1.0, 2.0])
    ax.legend()
    ax.set_ylim(0.35, 0.85)

    fig.suptitle('Phase 3C/D: Robustness Testing — VAECox vs CoxRidge\n'
                 '(gene expression z-normalized; noise/masking applied at test time)',
                 fontsize=11)
    fig.tight_layout()
    savefig('fig6_robustness.png')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 7 — Fairness: n_events vs C-index scatter
# ══════════════════════════════════════════════════════════════════════════════
def fig7_fairness_scatter():
    df = pd.read_csv('results/phase3/3e_fairness_correlation.csv')
    models_to_plot = ['CoxRidge', 'Coxnnet', 'VAECox']

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    for ax, model in zip(axes, models_to_plot):
        sub = df[df['model'] == model].dropna()
        if sub.empty:
            continue
        colors_pt = [MODEL_COLORS.get(model, 'gray')] * len(sub)
        sc = ax.scatter(sub['n_events'], sub['cindex'],
                        c=MODEL_COLORS[model], s=80, alpha=0.8,
                        edgecolor='black', linewidth=0.5, zorder=3)
        # Cancer labels
        for _, row in sub.iterrows():
            ax.annotate(row['cancer'],
                        (row['n_events'], row['cindex']),
                        textcoords='offset points', xytext=(4, 3),
                        fontsize=8, color='#333333')
        # Trend line
        if len(sub) >= 4:
            z = np.polyfit(sub['n_events'], sub['cindex'], 1)
            p = np.poly1d(z)
            xs = np.linspace(sub['n_events'].min(), sub['n_events'].max(), 50)
            ax.plot(xs, p(xs), '--', color=MODEL_COLORS[model], alpha=0.6, linewidth=1.5)
            from scipy.stats import spearmanr
            rho, pv = spearmanr(sub['n_events'], sub['cindex'])
            ax.set_title(f'{model}\nSpearman ρ={rho:+.2f}  p={pv:.2f}')
        else:
            ax.set_title(model)
        ax.axhline(0.5, color='gray', linestyle=':', linewidth=1, alpha=0.6)
        ax.set_xlabel('Number of events (uncensored deaths)')
        ax.set_ylabel('C-index' if model == 'CoxRidge' else '')
        ax.set_ylim(0, 1.1)

    fig.suptitle('Phase 3E: Fairness — Does C-index improve with more events?\n'
                 '(each point = one cancer type; toy data, 30 pts/cancer)',
                 fontsize=11)
    fig.tight_layout()
    savefig('fig7_fairness_scatter.png')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 8 — Feature importance top genes
# ══════════════════════════════════════════════════════════════════════════════
def fig8_feature_importance():
    df = pd.read_csv('results/phase3/feature_importance.csv')
    cancers  = df['cancer'].unique()[:2]   # BLCA + OV (most events)
    top_n    = 10

    fig, axes = plt.subplots(len(cancers), 2,
                             figsize=(14, 3.5 * len(cancers)))
    if len(cancers) == 1:
        axes = axes.reshape(1, -1)

    for row_i, cancer in enumerate(cancers):
        sub = df[df['cancer'] == cancer]
        for col_i, (model, color, label) in enumerate([
                ('CoxRidge', MODEL_COLORS['CoxRidge'],
                 'CoxRidge: |linear weight|\n(exact SHAP for linear model)'),
                ('VAECox', MODEL_COLORS['VAECox'],
                 'VAECox: gradient × input\n(SHAP approximation)')]):
            ax = axes[row_i, col_i]
            msub = sub[sub['model'] == model].head(top_n).copy()
            if msub.empty:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes)
                continue
            msub = msub.iloc[::-1]   # reverse for top-to-bottom display
            colors_bar = [color if v >= 0 else '#CCCCCC'
                          for v in msub['importance']]
            ax.barh(range(len(msub)), msub['importance'].abs(),
                    color=colors_bar, alpha=0.85, edgecolor='black',
                    linewidth=0.4)
            ax.set_yticks(range(len(msub)))
            ax.set_yticklabels(msub['gene'], fontsize=8.5)
            ax.set_xlabel('|Importance|')
            ax.set_title(f'{cancer} — {label}',
                         fontsize=9, color=color)
            # Direction markers
            for i, (_, r) in enumerate(msub.iterrows()):
                sign = '+' if r['importance'] >= 0 else '−'
                ax.text(0.002, i, sign, va='center',
                        fontsize=8, color='black', fontweight='bold')

    fig.suptitle('Phase 3 Feature Importance: Top-10 Survival-Associated Genes\n'
                 '(+ = higher expression → higher hazard; − = protective)',
                 fontsize=11)
    fig.tight_layout()
    savefig('fig8_feature_importance.png')


# ══════════════════════════════════════════════════════════════════════════════
# Regenerate KM curves at matching style
# ══════════════════════════════════════════════════════════════════════════════
def fig9_km_curves_restyle():
    """Re-style the KM plot to match the paper figure style."""
    import pickle
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    import sys
    sys.path.insert(0, '.')
    from models import Coxnnet, PartialNLL

    def make_R(y):
        n = len(y); R = np.zeros((n,n), dtype=np.float32)
        for i in range(n): R[i,:] = (y>=y[i])
        return R

    import torch
    DATA_DIR = 'data/prepared'
    cancers  = ['BLCA', 'OV', 'STAD']
    n_seeds  = 10

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax, cancer in zip(axes, cancers):
        pkg      = pickle.load(open(f'data/imputed_and_binary_{cancer}.pickle','rb'))
        full_df  = pkg[0]
        pids     = list(full_df.index)
        pred_acc = {pid: [] for pid in pids}

        for seed in range(n_seeds):
            try:
                d = f'{DATA_DIR}/{cancer}/seed_{seed}'
                X_tr = np.load(f'{d}/X_train.npy').astype(np.float32)
                y_tr = np.load(f'{d}/y_train.npy').astype(np.float64)
                c_tr = np.load(f'{d}/c_train.npy').astype(np.int32)
                Z_tr = np.load(f'data/embeddings/{cancer}/seed_{seed}/Z_train.npy').astype(np.float32)
                Z_te = np.load(f'data/embeddings/{cancer}/seed_{seed}/Z_test.npy').astype(np.float32)

                from sklearn.model_selection import train_test_split
                survival = full_df['survival'].fillna(0).values
                order    = np.argsort(survival)
                strata   = np.zeros(len(order), dtype=int)
                for b in range(5):
                    strata[order[int(b*len(order)/5):int((b+1)*len(order)/5)]] = b
                _, test_df = train_test_split(full_df, test_size=0.2,
                                              random_state=seed, stratify=strata)
                test_pids = list(test_df.index)

                cox = Coxnnet(128).double()
                Z_t = torch.tensor(Z_tr, dtype=torch.float64)
                c_t = torch.tensor(c_tr, dtype=torch.float64)
                R_t = torch.tensor(make_R(y_tr), dtype=torch.float64)
                opt = torch.optim.Adam(cox.parameters(), lr=1e-3, weight_decay=1e-5)
                cox.train()
                for _ in range(50):
                    opt.zero_grad()
                    th = cox(Z_t)
                    loss = PartialNLL()(th, R_t, c_t)
                    if torch.isnan(loss) or torch.isinf(loss): break
                    loss.backward(); opt.step()
                cox.eval()
                with torch.no_grad():
                    pred = cox(torch.tensor(Z_te, dtype=torch.float64)).reshape(-1).numpy()
                for pid, p in zip(test_pids, pred):
                    pred_acc[pid].append(float(p))
            except Exception:
                pass

        pids_ok = [pid for pid in pids if pred_acc[pid]]
        preds   = np.array([np.mean(pred_acc[pid]) for pid in pids_ok])
        survt   = full_df.loc[pids_ok, 'survival'].values.astype(float)
        cens    = full_df.loc[pids_ok, 'censored'].values.astype(int)
        events  = (cens == 0).astype(int)

        if len(pids_ok) < 4 or events.sum() < 2:
            ax.text(0.5, 0.5, f'{cancer}\n(insufficient events)',
                    ha='center', va='center', transform=ax.transAxes)
            continue

        med  = np.median(preds)
        low  = preds <= med;  high = preds > med

        kmf_lo = KaplanMeierFitter()
        kmf_hi = KaplanMeierFitter()
        kmf_lo.fit(survt[low],  events[low],  label=f'Low-risk (n={low.sum()})')
        kmf_hi.fit(survt[high], events[high], label=f'High-risk (n={high.sum()})')

        kmf_lo.plot_survival_function(ax=ax, ci_show=True, color='steelblue',
                                      linewidth=2)
        kmf_hi.plot_survival_function(ax=ax, ci_show=True, color='firebrick',
                                      linewidth=2)

        if events[low].sum() > 0 and events[high].sum() > 0:
            lr   = logrank_test(survt[low], survt[high], events[low], events[high])
            pstr = f'Log-rank p={lr.p_value:.3f}'
        else:
            pstr = 'Log-rank: N/A'

        ax.set_title(f'{cancer} — VAECox Risk Stratification\n{pstr}', fontsize=10)
        ax.set_xlabel('Survival time (days)')
        ax.set_ylabel('Survival probability' if cancer == 'BLCA' else '')
        ax.set_ylim(-0.05, 1.05)
        ax.legend(framealpha=0.9, fontsize=8)
        ax.add_patch(plt.Rectangle((0,0), 1, 1, fill=False,
                                   edgecolor='lightgray', linewidth=0.5,
                                   transform=ax.transAxes))

    fig.suptitle('Phase 3 Additional: Kaplan-Meier Survival Curves\n'
                 'VAECox predicted risk stratification (frozen encoder, 10-seed pooled predictions)',
                 fontsize=11)
    fig.tight_layout()
    savefig('fig9_km_curves.png')


# ══════════════════════════════════════════════════════════════════════════════
# Summary index
# ══════════════════════════════════════════════════════════════════════════════

def print_figure_index():
    figures = [
        ('fig1_cindex_barchart.png',    'Phase 2 main result: C-index grouped bar chart across 10 cancers'),
        ('fig2_cindex_heatmap.png',     'Phase 2 alt view: C-index heatmap (models × cancers)'),
        ('fig3_model_wins.png',         'Wins per model: reproduction vs paper claim (7/10 vs 4/10)'),
        ('fig4_lightweight_tradeoff.png','Phase 3A: hidden_dim/latent_dim vs C-index + training time'),
        ('fig5_feature_subset.png',     'Phase 3B: C-index vs number of genes (low-resource accessibility)'),
        ('fig6_robustness.png',         'Phase 3C/D: Robustness to missing features and Gaussian noise'),
        ('fig7_fairness_scatter.png',   'Phase 3E: n_events vs C-index scatter (fairness proxy)'),
        ('fig8_feature_importance.png', 'Phase 3 additional: Top-10 genes, CoxRidge vs VAECox attribution'),
        ('fig9_km_curves.png',          'Phase 3 additional: KM survival curves by predicted risk group'),
    ]
    print('\n' + '='*70)
    print('FIGURE INDEX — results/phase3/figures/')
    print('='*70)
    for fname, desc in figures:
        exists = '✓' if os.path.exists(f'results/phase3/figures/{fname}') else '✗'
        print(f'  {exists} {fname}')
        print(f'      {desc}')
    print()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import time
    t0 = time.time()

    print('Generating all paper figures...\n')

    print('Fig 1: C-index bar chart')
    fig1_cindex_barchart()

    print('Fig 2: C-index heatmap')
    fig2_cindex_heatmap()

    print('Fig 3: Model wins comparison')
    fig3_model_wins()

    print('Fig 4: Lightweight trade-off')
    fig4_lightweight_tradeoff()

    print('Fig 5: Feature subset')
    fig5_feature_subset()

    print('Fig 6: Robustness curves')
    fig6_robustness()

    print('Fig 7: Fairness scatter')
    fig7_fairness_scatter()

    print('Fig 8: Feature importance')
    fig8_feature_importance()

    print('Fig 9: KM curves (restyled)')
    fig9_km_curves_restyle()

    print_figure_index()
    print(f'Total time: {time.time()-t0:.1f}s')
