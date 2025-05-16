import scanpy as sc
import pandas as pd
import os

# === Config ===
adata_path = "summary_plots/qc_filtered_unstripped/adata_qc_filtered_clustered.h5ad"
output_dir = "cluster_qc_summary_striped"
os.makedirs(output_dir, exist_ok=True)

# === Load data
adata = sc.read(adata_path)

# === 1. Cell count per cluster
cluster_counts = adata.obs["leiden_qc"].value_counts().sort_index()
cluster_counts_df = cluster_counts.reset_index()
cluster_counts_df.columns = ["cluster", "total_cells"]
cluster_counts_df.to_csv(os.path.join(output_dir, "cell_count_per_cluster.csv"), index=False)
print("✅ Saved: cell_count_per_cluster.csv")

# === 2. Breakdown by primary/secondary (fixing index issue)
if "labels_joint_source" in adata.obs.columns:
    subtype_counts = (
        adata.obs.groupby(["leiden_qc", "labels_joint_source"], observed=True).size()
        .unstack(fill_value=0)
        .rename(columns={"primary": "primary_cells", "secondary": "secondary_cells"})
    )
    subtype_counts.index.name = "cluster"
    subtype_counts = subtype_counts.reset_index()

    summary_df = pd.merge(cluster_counts_df, subtype_counts, on="cluster", how="left")
    summary_df.to_csv(os.path.join(output_dir, "cell_count_per_cluster_with_subtypes.csv"), index=False)
    print("✅ Saved: cell_count_per_cluster_with_subtypes.csv")
else:
    summary_df = cluster_counts_df.copy()
    print("⚠️ labels_joint_source not found, skipping subtype summary.")

# === 3. Overall summary
n_total = adata.n_obs
n_primary = (adata.obs["labels_joint_source"] == "primary").sum()
n_secondary = (adata.obs["labels_joint_source"] == "secondary").sum()

with open(os.path.join(output_dir, "summary_counts.txt"), "w") as f:
    f.write(f"📦 Total cells: {n_total}\n")
    f.write(f"🔍 Primary cells: {n_primary}\n")
    f.write(f"🔍 Secondary cells: {n_secondary}\n")
    f.write(f"🧮 Number of clusters: {adata.obs['leiden_qc'].nunique()}\n")

print("✅ Saved: summary_counts.txt")

