import scanpy as sc
import matplotlib.pyplot as plt
import os

# === Config
adata_path = "summary_plots/qc_filtered_unstripped/adata_qc_filtered_clustered.h5ad"
output_dir = "cluster7_pacemaker_marker_umaps"
os.makedirs(output_dir, exist_ok=True)
cluster_id = "7"

# === List of SA node/pacemaker markers
pacemaker_markers = [
    "Vsnl1", "Hcn4", "Tbx3", "Shox2", "Isl1", "Tbx18", "Gja5", "Gjc1", "Kcnj3"
]

# === Load and subset
adata = sc.read(adata_path)
adata_7 = adata[adata.obs["leiden_qc"] == cluster_id].copy()

# === Plot UMAPs for each gene
for gene in pacemaker_markers:
    if gene not in adata_7.var_names:
        print(f"⚠️ {gene} not found in adata.var_names — skipping")
        continue

    sc.pl.umap(
        adata_7,
        color=gene,
        vmin='p1',
        vmax='p99.5',
        cmap="viridis",
        title=f"Cluster 7 — {gene}",
        show=False
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{gene}_umap.png"), dpi=300)
    plt.close()

print("✅ Pacemaker gene UMAPs for Cluster 7 saved.")

