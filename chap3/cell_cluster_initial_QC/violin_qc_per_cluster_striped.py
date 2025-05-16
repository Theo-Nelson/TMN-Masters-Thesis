import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Config ===
adata_path = "summary_plots/qc_filtered_unstripped/adata_qc_filtered_clustered.h5ad"
output_dir = "cluster_qc_summary_striped"
os.makedirs(output_dir, exist_ok=True)

# === Load data
adata = sc.read(adata_path)

# === Verify necessary columns
required_cols = ["leiden_qc", "total_counts", "n_genes_by_counts", "bin_count"]
missing = [col for col in required_cols if col not in adata.obs.columns]
if missing:
    raise ValueError(f"Missing required obs columns: {missing}")

# === Metrics to plot
metrics = {
    "total_counts": "UMI Count",
    "n_genes_by_counts": "Gene Count",
    "bin_count": "Bin Count"
}

# === Violin plots by cluster
for metric, label in metrics.items():
    plt.figure(figsize=(12, 6))
    sns.violinplot(
        data=adata.obs,
        x="leiden_qc",
        y=metric,
        scale="width",
        inner="quartile",
        cut=0
    )
    plt.title(f"{label} per Cluster")
    plt.xlabel("Leiden Cluster")
    plt.ylabel(label)
    plt.xticks(rotation=90)
    plt.tight_layout()
    fname = f"violin_{metric}_by_cluster.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=300)
    plt.close()
    print(f"✅ Saved violin plot: {fname}")

