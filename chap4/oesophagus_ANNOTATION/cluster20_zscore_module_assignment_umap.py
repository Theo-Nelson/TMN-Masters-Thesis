import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# === Config
adata_path = "summary_plots/qc_filtered_unstripped/adata_qc_filtered_clustered.h5ad"
module1_path = "cluster_20_zscore_heatmap/module1_genes.csv"
module2_path = "cluster_20_zscore_heatmap/module2_genes.csv"
output_dir = "cluster_20_zscore_module_assignment_umap"
os.makedirs(output_dir, exist_ok=True)

# === Load clustered object and subset Cluster 20
adata = sc.read(adata_path)
adata_20 = adata[adata.obs["leiden_qc"] == "20"].copy()

# === Load module gene lists
module1_genes = pd.read_csv(module1_path, header=None)[0].tolist()
module2_genes = pd.read_csv(module2_path, header=None)[0].tolist()

# Filter to genes present in adata
module1_genes = [g for g in module1_genes if g in adata_20.var_names]
module2_genes = [g for g in module2_genes if g in adata_20.var_names]

print(f"🧬 Module 1 genes: {len(module1_genes)}")
print(f"🧬 Module 2 genes: {len(module2_genes)}")

# === Compute per-cell module scores
adata_20.obs["module1_score"] = adata_20[:, module1_genes].X.mean(axis=1)
adata_20.obs["module2_score"] = adata_20[:, module2_genes].X.mean(axis=1)
adata_20.obs["module_score_diff"] = adata_20.obs["module1_score"] - adata_20.obs["module2_score"]

# === Assign cells to modules
def assign_module(row, threshold=0.1):
    if abs(row["module_score_diff"]) < threshold:
        return "Neither"
    return "Module_1" if row["module_score_diff"] > 0 else "Module_2"

adata_20.obs["coexpression_module"] = adata_20.obs.apply(assign_module, axis=1)

# === Plot UMAP by module assignment
sc.pl.umap(
    adata_20,
    color="coexpression_module",
    title="Cluster 20 — Z-score Module Assignment (UMAP)",
    palette=["#1f77b4", "#ff7f0e", "gray"],
    show=False
)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "umap_module_labels.png"), dpi=300)
plt.close()

# === Optional: also plot continuous module score difference
sc.pl.umap(
    adata_20,
    color="module_score_diff",
    cmap="coolwarm",
    title="Cluster 20 — Module Score Difference",
    show=False
)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "umap_module_score_diff.png"), dpi=300)
plt.close()

print("✅ UMAP plots with z-score module assignments saved.")

