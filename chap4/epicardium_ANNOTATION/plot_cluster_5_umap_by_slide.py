import scanpy as sc
import matplotlib.pyplot as plt
import os

# === Config
adata_path = "summary_plots/qc_filtered_unstripped/adata_qc_filtered_clustered.h5ad"
output_dir = "cluster_5_markers_umap"
os.makedirs(output_dir, exist_ok=True)

# === Load and subset
adata = sc.read(adata_path)
adata_5 = adata[adata.obs["leiden_qc"] == "5"].copy()

# === Plot UMAP colored by sample (slide)
sc.pl.umap(
    adata_5,
    color="sample",
    title="Cluster 5 — Slide of Origin",
    show=False
)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "umap_cluster5_by_sample.png"), dpi=300)
plt.close()

print("✅ UMAP of Cluster 5 colored by slide saved to:", os.path.join(output_dir, "umap_cluster5_by_sample.png"))

