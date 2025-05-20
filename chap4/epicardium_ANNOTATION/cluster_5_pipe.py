import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.cluster.hierarchy import linkage, leaves_list, cut_tree

# === Config
cluster_id = "5"
samples = ["WT_A1", "WT_D1"]
adata_path = "summary_plots/qc_filtered_unstripped/adata_qc_filtered_clustered.h5ad"
coexpr_csv = f"cluster_marker_coexpression_all/cluster_{cluster_id}/normalized_matrix.csv"
base_dir = "bin2cell_output_unstripped"
mpp = 0.2
img_key = f"{mpp}_mpp_150_buffer"
basis = "spatial_cropped_150_buffer"

output_dir = f"cluster_{cluster_id}_zscore_module_analysis"
heatmap_dir = os.path.join(output_dir, "zscore_heatmap")
spatial_dir = os.path.join(output_dir, "spatial")
os.makedirs(heatmap_dir, exist_ok=True)
os.makedirs(spatial_dir, exist_ok=True)

# === Load AnnData and subset
adata = sc.read(adata_path)
adata_cluster = adata[adata.obs["leiden_qc"] == cluster_id].copy()
adata_cluster.obs["array_row_r"] = adata_cluster.obs["array_row"].round(2)
adata_cluster.obs["array_col_r"] = adata_cluster.obs["array_col"].round(2)

# === Load coexpression matrix
coexpr_df = pd.read_csv(coexpr_csv, index_col=0)
genes = coexpr_df.index.tolist()

# === Z-score normalize rows (excluding diagonal)
np.fill_diagonal(coexpr_df.values, np.nan)
row_means = coexpr_df.mean(axis=1).to_numpy().reshape(-1, 1)
row_stds = coexpr_df.std(axis=1).to_numpy().reshape(-1, 1) + 1e-8
zscore_matrix = (coexpr_df.to_numpy() - row_means) / row_stds
zscore_df = pd.DataFrame(zscore_matrix, index=genes, columns=genes)
np.fill_diagonal(zscore_df.values, 0.0)

# === Cluster rows
link = linkage(zscore_df.values, method="average")
ordered_idx = leaves_list(link)
ordered_genes = [genes[i] for i in ordered_idx]
zscore_df_ordered = zscore_df.loc[ordered_genes, ordered_genes]

# === Cut into 2 modules
gene_clusters = cut_tree(link, n_clusters=2).flatten()
gene_map = dict(zip(genes, gene_clusters))
module1_genes = [g for g, c in gene_map.items() if c == 0 and g in adata_cluster.var_names]
module2_genes = [g for g, c in gene_map.items() if c == 1 and g in adata_cluster.var_names]

pd.Series(module1_genes).to_csv(os.path.join(heatmap_dir, "module1_genes.csv"), index=False)
pd.Series(module2_genes).to_csv(os.path.join(heatmap_dir, "module2_genes.csv"), index=False)

# === Save z-score heatmap
plt.figure(figsize=(20, 20))
sns.heatmap(zscore_df_ordered, cmap="vlag", center=0, xticklabels=True, yticklabels=True)
plt.title(f"Cluster {cluster_id} — Z-score Coexpression Heatmap")
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)
plt.tight_layout()
plt.savefig(os.path.join(heatmap_dir, "zscore_coexpression_heatmap.png"), dpi=300)
plt.close()

# === Score cells and assign module
adata_cluster.obs["module1_score"] = adata_cluster[:, module1_genes].X.mean(axis=1)
adata_cluster.obs["module2_score"] = adata_cluster[:, module2_genes].X.mean(axis=1)
adata_cluster.obs["module_score_diff"] = adata_cluster.obs["module1_score"] - adata_cluster.obs["module2_score"]

def assign_module(row, threshold=0.1):
    if abs(row["module_score_diff"]) < threshold:
        return "Neither"
    return "Module_1" if row["module_score_diff"] > 0 else "Module_2"

adata_cluster.obs["coexpression_module"] = adata_cluster.obs.apply(assign_module, axis=1)

# === UMAP plot
sc.pl.umap(
    adata_cluster,
    color="coexpression_module",
    title=f"Cluster {cluster_id} — Module Assignment (UMAP)",
    palette=["#1f77b4", "#ff7f0e", "lightgray"],
    show=False
)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "umap_module_labels.png"), dpi=300)
plt.close()

# === Spatial plots per sample
adata.obs["array_row_r"] = adata.obs["array_row"].round(2)
adata.obs["array_col_r"] = adata.obs["array_col"].round(2)

for sample_id in samples:
    print(f"\n🧬 Spatial plot for {sample_id}")
    sample_path = f"{base_dir}/{sample_id}/{sample_id}_cells.h5ad"
    adata_sample = sc.read(sample_path)

    if "spatial" in adata_sample.uns and sample_id not in adata_sample.uns["spatial"]:
        old_key = list(adata_sample.uns["spatial"].keys())[0]
        adata_sample.uns["spatial"][sample_id] = adata_sample.uns["spatial"].pop(old_key)

    adata_sample.obs["array_row_r"] = adata_sample.obs["array_row"].round(2)
    adata_sample.obs["array_col_r"] = adata_sample.obs["array_col"].round(2)

    merged_df = adata_cluster[adata_cluster.obs["sample"] == sample_id].obs[
        ["array_row_r", "array_col_r", "coexpression_module"]
    ]
    merged = pd.merge(
        adata_sample.obs.reset_index(),
        merged_df,
        on=["array_row_r", "array_col_r"],
        how="left"
    ).drop_duplicates(subset="index", keep="first").set_index("index")

    # Ensure NA is a valid category
    categories = ["Module_1", "Module_2", "Neither", "NA"]
    merged["coexpression_module"] = merged["coexpression_module"].astype("category")
    merged["coexpression_module"] = merged["coexpression_module"].cat.set_categories(categories)
    adata_sample.obs["coexpression_module"] = merged["coexpression_module"].fillna("NA")

    sc.pl.spatial(
        adata_sample,
        color="coexpression_module",
        img_key=img_key,
        basis=basis,
        title=f"{sample_id} — Cluster {cluster_id} Modules",
        palette=["#1f77b4", "#ff7f0e", "green", "lightgray"],
        show=False
    )
    plt.tight_layout()
    plt.savefig(os.path.join(spatial_dir, f"{sample_id}_module_labels.png"), dpi=900)
    plt.close()

print("\n✅ Cluster 5: heatmap, UMAP, and spatial module plots saved.")

