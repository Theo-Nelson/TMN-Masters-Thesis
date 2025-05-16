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

# === Count cells per cluster
cluster_counts = adata.obs["leiden_qc"].value_counts().sort_index()
cluster_counts_df = cluster_counts.reset_index()
cluster_counts_df.columns = ["cluster", "cell_count"]

# === Plot
plt.figure(figsize=(12, 6))
sns.barplot(data=cluster_counts_df, x="cluster", y="cell_count", color="skyblue")
plt.title("Number of Cells per Cluster (Striped Dataset)")
plt.xlabel("Leiden Cluster")
plt.ylabel("Cell Count")
plt.xticks(rotation=90)
plt.tight_layout()
plot_path = os.path.join(output_dir, "barplot_cells_per_cluster.png")
plt.savefig(plot_path, dpi=300)
plt.close()

print(f"✅ Saved bar plot to: {plot_path}")

