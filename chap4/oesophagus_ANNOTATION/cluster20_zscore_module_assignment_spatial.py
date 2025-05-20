import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import os

# === Config
samples = ["WT_A1", "WT_D1"]
cluster_id = "20"
mpp = 0.2
img_key = f"{mpp}_mpp_150_buffer"
basis = "spatial_cropped_150_buffer"

# === Paths
combined_path = "summary_plots/qc_filtered_unstripped/adata_qc_filtered_clustered.h5ad"
module1_path = "cluster_20_zscore_heatmap/module1_genes.csv"
module2_path = "cluster_20_zscore_heatmap/module2_genes.csv"
base_dir = "bin2cell_output_unstripped"
output_dir = "cluster_20_zscore_module_assignment_spatial"
os.makedirs(output_dir, exist_ok=True)

# === Load combined object and cluster 20 subset
adata_combined = sc.read(combined_path)
adata_combined.obs["array_row_r"] = adata_combined.obs["array_row"].round(2)
adata_combined.obs["array_col_r"] = adata_combined.obs["array_col"].round(2)
adata_20 = adata_combined[adata_combined.obs["leiden_qc"] == cluster_id].copy()
adata_20.obs["array_row_r"] = adata_20.obs["array_row"].round(2)
adata_20.obs["array_col_r"] = adata_20.obs["array_col"].round(2)

# === Load module gene lists
module1_genes = pd.read_csv(module1_path, header=None)[0].tolist()
module2_genes = pd.read_csv(module2_path, header=None)[0].tolist()
module1_genes = [g for g in module1_genes if g in adata_20.var_names]
module2_genes = [g for g in module2_genes if g in adata_20.var_names]

# === Score cells
adata_20.obs["module1_score"] = adata_20[:, module1_genes].X.mean(axis=1)
adata_20.obs["module2_score"] = adata_20[:, module2_genes].X.mean(axis=1)

def assign_module(row, threshold=0.1):
    if abs(row["module1_score"] - row["module2_score"]) < threshold:
        return "Neither"
    return "Module_1" if row["module1_score"] > row["module2_score"] else "Module_2"

adata_20.obs["coexpression_module"] = adata_20.obs.apply(assign_module, axis=1)

# === Spatial plot per sample
for sample_id in samples:
    print(f"\n🧬 Plotting for {sample_id}...")
    path = f"{base_dir}/{sample_id}/{sample_id}_cells.h5ad"
    plot_dir = os.path.join(output_dir, sample_id)
    os.makedirs(plot_dir, exist_ok=True)

    adata_sample = sc.read(path)

    if "spatial" in adata_sample.uns and sample_id not in adata_sample.uns["spatial"]:
        old_key = list(adata_sample.uns["spatial"].keys())[0]
        adata_sample.uns["spatial"][sample_id] = adata_sample.uns["spatial"].pop(old_key)

    adata_sample.obs["array_row_r"] = adata_sample.obs["array_row"].round(2)
    adata_sample.obs["array_col_r"] = adata_sample.obs["array_col"].round(2)

    # Merge module labels for cluster 20 cells
    merged_df = adata_20[adata_20.obs["sample"] == sample_id].obs[
        ["array_row_r", "array_col_r", "coexpression_module"]
    ]
    merged = pd.merge(
        adata_sample.obs.reset_index(),
        merged_df,
        on=["array_row_r", "array_col_r"],
        how="left"
    ).drop_duplicates(subset="index", keep="first").set_index("index")

    adata_sample.obs["coexpression_module"] = merged["coexpression_module"].fillna("NA").astype("category")

    # Plot
    sc.pl.spatial(
        adata_sample,
        color="coexpression_module",
        img_key=img_key,
        basis=basis,
        title=f"{sample_id} — Cluster 20 Module Assignment (z-score)",
        palette=["#1f77b4", "#ff7f0e", "lightgray","green"],
        show=False
    )
    out_path = os.path.join(plot_dir, f"{sample_id}_module_labels.png")
    plt.savefig(out_path, dpi=900, bbox_inches="tight")
    plt.close()

print("\n✅ Spatial module assignment plots (z-score-based) saved.")

