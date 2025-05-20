import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import os

# === Config
adata_path = "summary_plots/qc_filtered_unstripped/adata_qc_filtered_clustered.h5ad"
marker_csv = "summary_plots/qc_filtered_unstripped/top_markers/top100_markers_cluster_5.csv"
output_dir = "cluster_5_markers_umap"
os.makedirs(output_dir, exist_ok=True)
cluster_id = "5"
n_genes = 100

# === Load data
adata = sc.read(adata_path)

# === Subset to cluster 5
adata_5 = adata[adata.obs["leiden_qc"] == cluster_id].copy()

# === Load top marker genes for cluster 5
df_markers = pd.read_csv(marker_csv)
top_genes = df_markers["names"].head(n_genes).tolist()
top_genes = [g for g in top_genes if g in adata.var_names]  # keep only valid genes

# === UMAP plots for top genes
for gene in top_genes:
    sc.pl.umap(
        adata_5,
        color=gene,
        vmin='p1',
        vmax='p99.5',
        title=f"Cluster 5 — {gene}",
        show=False
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"cluster_5_umap_{gene}.png"), dpi=300)
    plt.close()

print(f"✅ UMAPs for top {n_genes} marker genes of Cluster 5 saved to: {output_dir}")

