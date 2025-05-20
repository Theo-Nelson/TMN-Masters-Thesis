import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.cluster.hierarchy import linkage, leaves_list, cut_tree

# === Config
matrix_csv = "cluster_marker_coexpression_all/cluster_20/normalized_matrix.csv"
output_dir = "cluster_20_zscore_heatmap"
os.makedirs(output_dir, exist_ok=True)

# === Load co-expression matrix
df = pd.read_csv(matrix_csv, index_col=0)

# === Set diagonal to NaN to exclude from z-score calc
np.fill_diagonal(df.values, np.nan)

# === Row-wise z-score (excluding diagonal from mean/std)
row_means = df.mean(axis=1).to_numpy().reshape(-1, 1)
row_stds = df.std(axis=1).to_numpy().reshape(-1, 1) + 1e-8
zscore_matrix = (df.to_numpy() - row_means) / row_stds
zscore_df = pd.DataFrame(zscore_matrix, index=df.index, columns=df.columns)

# === Restore diagonal to 0 for heatmap clarity
np.fill_diagonal(zscore_df.values, 0.0)

# === Hierarchical clustering on z-scored rows
link = linkage(zscore_df.values, method="average", metric="euclidean")
ordered_idx = leaves_list(link)
ordered_genes = [df.index[i] for i in ordered_idx]
z_df_ordered = zscore_df.loc[ordered_genes, ordered_genes]

# === Cut into 2 gene modules using cut_tree
gene_clusters = cut_tree(link, n_clusters=2).flatten()
gene_map = dict(zip(df.index, gene_clusters))
module1_genes = [g for g, c in gene_map.items() if c == 0]
module2_genes = [g for g, c in gene_map.items() if c == 1]

# === Print and save module sizes
print(f"🧬 Module 1: {len(module1_genes)} genes")
print(f"🧬 Module 2: {len(module2_genes)} genes")

with open(os.path.join(output_dir, "module_gene_counts.txt"), "w") as f:
    f.write(f"Module 1: {len(module1_genes)} genes\n")
    f.write(f"Module 2: {len(module2_genes)} genes\n")

# === Save gene lists
pd.Series(module1_genes).to_csv(os.path.join(output_dir, "module1_genes.csv"), index=False)
pd.Series(module2_genes).to_csv(os.path.join(output_dir, "module2_genes.csv"), index=False)

# === Plot z-score heatmap
plt.figure(figsize=(20, 20))
sns.heatmap(
    z_df_ordered,
    cmap="vlag",
    center=0,
    xticklabels=True,
    yticklabels=True
)
plt.title("Cluster 20 — Z-score Normalized Co-expression Heatmap")
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "zscore_coexpression_heatmap.png"), dpi=300)
plt.close()

print(f"✅ Z-score heatmap and module gene lists saved to: {output_dir}")

