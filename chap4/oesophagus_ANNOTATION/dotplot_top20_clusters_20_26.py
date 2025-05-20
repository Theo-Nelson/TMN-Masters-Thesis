import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import os

# === Config
adata_path = "summary_plots/qc_filtered_unstripped/adata_qc_filtered_clustered.h5ad"
marker_dir = "summary_plots/qc_filtered_unstripped/top_markers"
output_dir = "cluster_20_26_marker_expression"
output_plot = os.path.join(output_dir, "dotplot_top20_cluster_20_26.png")
os.makedirs(output_dir, exist_ok=True)

# === Load adata
adata = sc.read(adata_path)

# === Load top 20 marker genes for cluster 20 and 26
df_20 = pd.read_csv(os.path.join(marker_dir, "top100_markers_cluster_20.csv")).head(20)
df_26 = pd.read_csv(os.path.join(marker_dir, "top100_markers_cluster_26.csv")).head(20)

# Combine and deduplicate genes
marker_genes = list(dict.fromkeys(df_20["names"].tolist() + df_26["names"].tolist()))
marker_genes = [g for g in marker_genes if g in adata.var_names]

# Subset to clusters 20 and 26 first
adata_subset = adata[adata.obs["leiden_qc"].isin(["20", "26"])].copy()

# Then generate the dotplot
sc.pl.dotplot(
    adata_subset,
    var_names=marker_genes,
    groupby="leiden_qc",
    standard_scale="var",
    swap_axes=False,
    show=False
)

plt.gcf().set_size_inches(12, 10)  # ⬅️ wider figure

plt.title("Relative Expr")
plt.tight_layout()
plt.savefig(output_plot, dpi=300)
plt.close()

print(f"✅ Dot plot saved to: {output_plot}")

