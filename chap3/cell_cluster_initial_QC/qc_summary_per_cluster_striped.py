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

# === Sanity check
required_cols = ["leiden_qc", "total_counts", "n_genes_by_counts", "bin_count"]
missing = [col for col in required_cols if col not in adata.obs.columns]
if missing:
    raise ValueError(f"Missing required obs columns: {missing}")

# === Summary stats per cluster
qc_stats = (
    adata.obs.groupby("leiden_qc")[["total_counts", "n_genes_by_counts", "bin_count"]]
    .agg(["mean", "std", "median", "count"])
)
qc_stats.columns = ["_".join(c) for c in qc_stats.columns]
qc_stats.index.name = "cluster"
qc_stats.reset_index(inplace=True)

# === Save as CSV
qc_stats.to_csv(os.path.join(output_dir, "qc_summary_per_cluster.csv"), index=False)
print(f"✅ Saved QC summary to: {output_dir}/qc_summary_per_cluster.csv")

# === Bar plots for each QC metric
metrics = {
    "total_counts_mean": "Mean UMI Count",
    "n_genes_by_counts_mean": "Mean Gene Count",
    "bin_count_mean": "Mean Bin Count"
}

for metric, ylabel in metrics.items():
    plt.figure(figsize=(10, 5))
    sns.barplot(data=qc_stats, x="cluster", y=metric)
    plt.ylabel(ylabel)
    plt.xlabel("Cluster")
    plt.title(f"{ylabel} per Cluster")
    plt.xticks(rotation=90)
    plt.tight_layout()
    fname = f"barplot_{metric}.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=300)
    plt.close()
    print(f"✅ Saved plot: {fname}")

