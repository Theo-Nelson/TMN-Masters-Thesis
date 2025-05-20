import scanpy as sc
import matplotlib.pyplot as plt
import os

# === Config
adata_path = "summary_plots/qc_filtered_unstripped/adata_qc_filtered_clustered.h5ad"
output_path = "cluster7_umap_by_sample.png"
cluster_id = "7"

# === Load data
adata = sc.read(adata_path)

# === Subset to Cluster 7
adata_7 = adata[adata.obs["leiden_qc"] == cluster_id].copy()

# === Plot UMAP colored by sample
sc.pl.umap(
    adata_7,
    color="sample",
    title="Cluster 7 — UMAP by Sample",
    show=False
)
plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.close()

print(f"✅ UMAP plot for Cluster 7 colored by sample saved to: {output_path}")

