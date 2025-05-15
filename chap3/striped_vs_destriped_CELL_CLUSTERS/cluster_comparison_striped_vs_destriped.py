import scanpy as sc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# === Config ===
adata_striped_path = "summary_plots/qc_filtered_unstripped/adata_qc_filtered_clustered.h5ad"
adata_destriped_path = "summary_plots/qc_filtered_recombined/adata_qc_filtered_clustered.h5ad"
output_dir = "cluster_comparison_striped_vs_destriped"
os.makedirs(output_dir, exist_ok=True)
distance_threshold_um = 10.0

# === Load data
adata_striped = sc.read(adata_striped_path)
adata_destriped = sc.read(adata_destriped_path)

# === Match shared cells
shared_cells_all = adata_striped.obs_names.intersection(adata_destriped.obs_names)
adata_striped_all = adata_striped[shared_cells_all]
adata_destriped_all = adata_destriped[shared_cells_all]

# === Spatial proximity filter (≤ 10 µm)
if "spatial" in adata_striped_all.obsm and "spatial" in adata_destriped_all.obsm:
    coords_striped = adata_striped_all.obsm["spatial"]
    coords_destriped = adata_destriped_all.obsm["spatial"]
    spatial_distance = np.linalg.norm(coords_striped - coords_destriped, axis=1)
    spatial_mask = spatial_distance <= distance_threshold_um

    adata_striped = adata_striped_all[spatial_mask]
    adata_destriped = adata_destriped_all[spatial_mask]
    spatial_distance_filtered = spatial_distance[spatial_mask]

    print(f"✅ Kept {spatial_mask.sum()} cells within {distance_threshold_um} µm")
else:
    raise ValueError("Missing `.obsm['spatial']` in one or both AnnData objects.")

# === Cluster contingency table
df = pd.DataFrame({
    "striped_cluster": adata_striped.obs["leiden_qc"].astype(str),
    "destriped_cluster": adata_destriped.obs["leiden_qc"].astype(str)
})
confusion = pd.crosstab(df["striped_cluster"], df["destriped_cluster"])

# === Row-normalized heatmap
confusion_row = confusion.div(confusion.sum(axis=1), axis=0) * 100
plt.figure(figsize=(20, 16))
sns.heatmap(confusion_row, annot=True, fmt=".1f", cmap="YlGnBu",
            cbar_kws={"label": "% of striped cluster"}, annot_kws={"fontsize": 6})
plt.title("Row-normalized % Heatmap (Striped → Destriped Clusters)")
plt.xlabel("Destriped cluster")
plt.ylabel("Striped cluster")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "heatmap_striped_to_destriped_row.png"), dpi=300)
plt.close()

# === Column-normalized heatmap
confusion_col = confusion.div(confusion.sum(axis=0), axis=1) * 100
plt.figure(figsize=(20, 16))
sns.heatmap(confusion_col, annot=True, fmt=".1f", cmap="YlOrBr",
            cbar_kws={"label": "% of destriped cluster"}, annot_kws={"fontsize": 6})
plt.title("Column-normalized % Heatmap (Destriped → Striped Clusters)")
plt.xlabel("Destriped cluster")
plt.ylabel("Striped cluster")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "heatmap_destriped_to_striped_col.png"), dpi=300)
plt.close()

# === Save raw tables
confusion.to_csv(os.path.join(output_dir, "cluster_confusion_counts.csv"))
confusion_row.to_csv(os.path.join(output_dir, "cluster_confusion_percent_row_normalized.csv"))
confusion_col.to_csv(os.path.join(output_dir, "cluster_confusion_percent_column_normalized.csv"))

# === Stats
n_shared_spatial = adata_striped.n_obs
n_total_striped = sc.read(adata_striped_path).n_obs
n_total_destriped = sc.read(adata_destriped_path).n_obs
n_lost_striped = n_total_striped - n_shared_spatial
n_lost_destriped = n_total_destriped - n_shared_spatial

# === Primary/Secondary shared
labels_s = adata_striped.obs.get("labels_joint_source", pd.Series("NA", index=adata_striped.obs_names))
labels_d = adata_destriped.obs.get("labels_joint_source", pd.Series("NA", index=adata_destriped.obs_names))
shared_primary = ((labels_s == "primary") & (labels_d == "primary")).sum()
shared_secondary = ((labels_s == "secondary") & (labels_d == "secondary")).sum()

# === Total primary/secondary from full datasets
adata_striped_full = sc.read(adata_striped_path)
adata_destriped_full = sc.read(adata_destriped_path)
total_primary_striped = (adata_striped_full.obs["labels_joint_source"] == "primary").sum()
total_secondary_striped = (adata_striped_full.obs["labels_joint_source"] == "secondary").sum()
total_primary_destriped = (adata_destriped_full.obs["labels_joint_source"] == "primary").sum()
total_secondary_destriped = (adata_destriped_full.obs["labels_joint_source"] == "secondary").sum()

# === Write stats
with open(os.path.join(output_dir, "cell_overlap_stats.txt"), "w") as f:
    f.write(f"🔗 Shared cells used in comparison (≤ {distance_threshold_um} µm): {n_shared_spatial}\n")
    f.write(f"📦 Total striped cells: {n_total_striped}\n")
    f.write(f"📦 Total destriped cells: {n_total_destriped}\n")
    f.write(f"❌ Striped-only cells excluded: {n_lost_striped}\n")
    f.write(f"❌ Destriped-only cells excluded: {n_lost_destriped}\n\n")

    f.write(f"🔍 Shared primary cells (within threshold): {shared_primary} / {min(total_primary_striped, total_primary_destriped)}\n")
    f.write(f"🔍 Shared secondary cells (within threshold): {shared_secondary} / {min(total_secondary_striped, total_secondary_destriped)}\n\n")

    f.write(f"📐 Avg Euclidean distance between matched spatial coords: {spatial_distance_filtered.mean():.2f} µm\n")
    f.write(f"📐 Max Euclidean distance among matched: {spatial_distance_filtered.max():.2f} µm\n")

print("✅ Cluster alignment + proximity-based filtering completed.")

