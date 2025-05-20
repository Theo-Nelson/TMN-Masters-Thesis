import scanpy as sc
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

# === Config
adata_path = "summary_plots/qc_filtered_unstripped/adata_qc_filtered_clustered.h5ad"
marker_dir = "summary_plots/qc_filtered_unstripped/top_markers"
output_dir = "cluster_marker_coexpression_all"
os.makedirs(output_dir, exist_ok=True)

# === Load full data
adata = sc.read(adata_path)
all_clusters = sorted(adata.obs["leiden_qc"].cat.categories)

for cluster_id in all_clusters:
    print(f"\n🔬 Processing cluster {cluster_id}")
    cluster_out = os.path.join(output_dir, f"cluster_{cluster_id}")
    os.makedirs(cluster_out, exist_ok=True)

    # Subset to cells in this cluster
    adata_c = adata[adata.obs["leiden_qc"] == cluster_id].copy()
    adata_c = adata_c.raw.to_adata()  # replace .X with .raw.X, and restore all genes
    sc.pp.normalize_total(adata_c, target_sum=1e4)
    sc.pp.log1p(adata_c)
    
    # Load top 100 marker genes
    marker_file = os.path.join(marker_dir, f"top100_markers_cluster_{cluster_id}.csv")
    if not os.path.exists(marker_file):
        print(f"⚠️ Skipping cluster {cluster_id}: marker file not found")
        continue
    
    marker_genes = pd.read_csv(marker_file)["names"].head(100).tolist()
    marker_genes = [g for g in marker_genes if g in adata_c.var_names]
    # Show how many marker genes are missing
    missing_genes = [g for g in marker_genes if g not in adata_c.var_names]
    print(f"Cluster {cluster_id}: {len(marker_genes)} total markers, {len(missing_genes)} not found")
    if missing_genes:
        print("Example missing genes:", missing_genes[:5])

    if len(marker_genes) < 5 or adata_c.n_obs < 5:
        print(f"⚠️ Skipping cluster {cluster_id}: too few genes or cells")
        continue

    # Extract expression
    X = adata_c[:, marker_genes].X.toarray() if hasattr(adata_c[:, marker_genes].X, "toarray") else adata_c[:, marker_genes].X
    expr = np.array(X)

    # Normalize by gene means
    mean_expr = expr.mean(axis=0)
    norm_expr = expr / (mean_expr + 1e-8)

    # Compute co-expression score matrix
    coexpr_score = np.dot(norm_expr.T, norm_expr)

    # Set diagonal to 0
    np.fill_diagonal(coexpr_score, 0.0)

    # Cluster based on off-diagonal distance
    dist_matrix = 1 - (coexpr_score / (coexpr_score.max() + 1e-8))
    np.fill_diagonal(dist_matrix, 0.0)
    try:
        dist_condensed = squareform(dist_matrix)
        link = linkage(dist_condensed, method="average")
        ordered_idx = leaves_list(link)
        ordered_genes = [marker_genes[i] for i in ordered_idx]
    except Exception as e:
        print(f"❌ Skipping cluster {cluster_id}: clustering failed ({str(e)})")
        continue

    # Save matrix
    coexpr_df = pd.DataFrame(coexpr_score, index=marker_genes, columns=marker_genes)
    coexpr_df_ordered = coexpr_df.loc[ordered_genes, ordered_genes]
    coexpr_df_ordered.to_csv(os.path.join(cluster_out, "normalized_matrix.csv"))

    # Plot heatmap
    plt.figure(figsize=(16, 12))  # increase size
    ax = sns.heatmap(
        coexpr_df_ordered,
        cmap="mako",
        xticklabels=True,
        yticklabels=True,
        rasterized=True
    )
    plt.title(f"Cluster {cluster_id} — Normalized Multiplicative Co-expression", fontsize=14)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(cluster_out, "heatmap_clustered.png"), dpi=300)
    plt.close()


print("\n✅ All cluster co-expression matrices and heatmaps saved.")

