import scanpy as sc
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from scipy.spatial import KDTree

# === Config ===
adata_path = "summary_plots/qc_filtered_unstripped/adata_qc_filtered_clustered.h5ad"
output_dir = "cluster_qc_summary_striped"
os.makedirs(output_dir, exist_ok=True)
spatial_key = "spatial"
k_neighbors = 10

# === Load
adata = sc.read(adata_path)
coords = adata.obsm[spatial_key]
clusters = adata.obs["leiden_qc"].astype(str)

# === Build KNN graph using KDTree
tree = KDTree(coords)
_, knn_indices = tree.query(coords, k=k_neighbors + 1)  # include self

# === Helper: compute Moran's I manually
def compute_moran_i(x, W):
    x = x.astype(float)
    x_mean = np.mean(x)
    x_dev = x - x_mean
    num = 0.0
    denom = np.sum(x_dev ** 2)
    for i in range(len(x)):
        for j in W[i]:
            num += x_dev[i] * x_dev[j]
    weight_sum = sum(len(neighbors) for neighbors in W)
    return (len(x) / weight_sum) * (num / denom) if denom != 0 else np.nan

# === Build W as list of neighbor indices (exclude self)
W = [list(neigh[1:]) for neigh in knn_indices]  # remove self at index 0

# === Run Moran’s I per cluster
results = []
for cluster in tqdm(sorted(clusters.unique()), desc="Computing Moran’s I"):
    x = (clusters == cluster).astype(int).values
    moran_i = compute_moran_i(x, W)
    results.append({
        "cluster": cluster,
        "moran_I": moran_i,
        "n_cells": int(np.sum(x))
    })

# === Save
df = pd.DataFrame(results)
df.to_csv(os.path.join(output_dir, "moran_i_per_cluster.csv"), index=False)
print("✅ Saved Moran’s I results to moran_i_per_cluster.csv")

