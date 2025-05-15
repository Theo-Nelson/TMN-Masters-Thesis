import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import os

# === Config ===
adata_path = "qc_summary_striped_vs_destriped/adata_joint_striped_destriped.h5ad"
output_dir = "qc_summary_striped_vs_destriped"
os.makedirs(output_dir, exist_ok=True)

# === Load data ===
adata = sc.read(adata_path)
umap_coords = adata.obsm["X_umap"]
sources = adata.obs["source"].values
print(f"✅ Loaded AnnData with shape: {adata.shape} and UMAP coords: {umap_coords.shape}")

# === Fit k-NN model on UMAP coordinates ===
max_k = 5
nbrs = NearestNeighbors(n_neighbors=max_k + 1).fit(umap_coords)  # +1 because the first neighbor is self
distances, indices = nbrs.kneighbors(umap_coords)  # distances: (n_cells, max_k+1)

# === Remove distance to self (first column) ===
distances = distances[:, 1:]  # (n_cells, max_k)

# === Compute average distances for k=1 to 5 per cell ===
avg_distances = {
    f"k_{k}": distances[:, :k].mean(axis=1)
    for k in range(1, max_k + 1)
}

# === Convert to DataFrame with source labels ===
df = pd.DataFrame(avg_distances)
df["source"] = sources

# === Group by source and compute mean across cells ===
summary_df = (
    df.groupby("source")
    .agg({f"k_{k}": "mean" for k in range(1, max_k + 1)})
    .reset_index()
    .melt(id_vars="source", var_name="k", value_name="avg_distance")
)

# Convert "k_1" → 1 for plotting
summary_df["k"] = summary_df["k"].str.replace("k_", "").astype(int)

# === Plot ===
plt.figure(figsize=(7, 5))
for source, group in summary_df.groupby("source"):
    plt.plot(group["k"], group["avg_distance"], marker="o", label=source)

plt.title("Average UMAP Distance to k Nearest Neighbors")
plt.xlabel("k (nearest neighbors)")
plt.ylabel("Average distance")
plt.xticks(range(1, max_k + 1))
plt.legend(title="Source")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "umap_avg_neighbor_distance.png"), dpi=300)
plt.close()

print("✅ Neighbor distance plot saved.")

