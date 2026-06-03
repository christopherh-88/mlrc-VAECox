import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm

CANCERS = ['BLCA', 'BRCA', 'CESC', 'COAD', 'GBM', 'HNSC', 'KIRC', 'KIRP',
           'LAML', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'OV', 'PAAD', 'PRAD',
           'READ', 'SKCM', 'STAD', 'THCA']

OUTPUT_TSV = './data/toyforVAE_811_mRNA@.tsv'
OUTPUT_CSV = './data/toyforVAE_mRNA@_binary.csv'

imputed_list = []

print("Loading cancer datasets...")
for c in tqdm(CANCERS):
    path = f'./data/imputed_and_binary_{c}.pickle'
    with open(path, 'rb') as f:
        package = pickle.load(f)
    df = package[0].copy()
    df = df.drop(columns=['censored', 'survival'])
    df.columns = ['mRNA@' + col for col in df.columns]
    imputed_list.append(df)

df_all = pd.concat(imputed_list, axis=0, sort=True)
df_all = df_all.fillna(0.0)
print(f"Combined shape: {df_all.shape}")

df_all = df_all.sample(frac=1.0, random_state=42)
n = len(df_all)
n_train = int(0.8 * n)
n_valid = int(0.9 * n)

fold = np.zeros(n, dtype=int)
fold[n_train:n_valid] = 1
fold[n_valid:] = 2
df_all['Fold@811'] = fold

print(f"Train: {(df_all['Fold@811']==0).sum()}, Valid: {(df_all['Fold@811']==1).sum()}, Test: {(df_all['Fold@811']==2).sum()}")

df_all.to_csv(OUTPUT_TSV, sep='\t', index_label='Samples')
print(f"Saved: {OUTPUT_TSV}")

gene_cols = [c for c in df_all.columns if c != 'Fold@811']
mask_df = pd.DataFrame(
    np.ones((len(df_all), len(gene_cols)), dtype=np.float32),
    index=df_all.index,
    columns=gene_cols
)
mask_df.to_csv(OUTPUT_CSV, sep=',', index_label='Samples')
print(f"Saved: {OUTPUT_CSV}")
