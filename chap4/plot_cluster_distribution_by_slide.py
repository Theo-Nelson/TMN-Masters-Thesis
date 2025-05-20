import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import os

# === Config
adata_path = "summary_plots/qc_filtered_unstripped/adata_qc_filtered_clustered.h5ad"
output_plot = "cluster_by_slide_distribution_stacked.png"

# === Load data
adata = sc.read(adata_path)

# === Create count table: cluster × sample
df = adata.obs[["sample", "leiden_qc"]].copy()
count_df = df.groupby(["leiden_qc", "sample"]).size().unstack(fill_value=0)

# === Normalize within each cluster (row-wise)
proportion_df = count_df.div(count_df.sum(axis=1), axis=0)

# === Sort clusters numerically
proportion_df.index = proportion_df.index.astype(int)
proportion_df = proportion_df.sort_index()

# === Plot as stacked bar chart
plt.figure(figsize=(16, 10))
ax = proportion_df.plot(
    kind="bar",
    stacked=True,
    colormap="tab20",
    edgecolor="black",
    width=0.7,
    ax=plt.gca()
)
plt.ylabel("Proportion of Cluster from Each Slide", fontsize=16)
plt.xlabel("Cluster (leiden_qc)", fontsize=16)
plt.title("Slide Composition per Cluster (Stacked Bar Plot)", fontsize=16)
ax.set_xticks(range(len(proportion_df.index)))
ax.set_xticklabels(proportion_df.index.astype(str), rotation=45, ha='center', fontsize=16)
plt.yticks(fontsize=16)
plt.legend(title="Sample", fontsize=14, title_fontsize=15, bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig(output_plot, dpi=300)
plt.close()


print(f"✅ Stacked cluster-by-slide distribution saved to: {output_plot}")

