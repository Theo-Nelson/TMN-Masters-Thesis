import os
import scanpy as sc
import matplotlib.pyplot as plt

# === Config
adata_path = "summary_plots/qc_filtered_unstripped/adata_qc_filtered_clustered.h5ad"
output_dir = "cluster_20_26_marker_expression"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "umap_cluster_20_26_only.png")

# === Load data
adata = sc.read(adata_path)

# === Subset to clusters 20 and 26
adata_subset = adata[adata.obs["leiden_qc"].isin(["20", "26"])].copy()

# === Plot UMAP
sc.pl.umap(
    adata_subset,
    color="leiden_qc",
    title="UMAP — Clusters 20 and 26",
    palette=["#1f77b4", "#ff7f0e"],
    show=False
)
plt.savefig(output_file, dpi=300, bbox_inches="tight")
plt.close()

print(f"✅ UMAP saved to: {output_file}")

