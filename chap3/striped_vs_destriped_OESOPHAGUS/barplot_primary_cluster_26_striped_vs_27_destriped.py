import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Config ===
adata_striped_path = "summary_plots/qc_filtered_unstripped_primary_only/adata_qc_filtered_clustered.h5ad"
adata_destriped_path = "summary_plots/qc_filtered_recombined_primary_only/adata_qc_filtered_clustered.h5ad"
output_dir = "cluster_comparison_striped_vs_destriped"
os.makedirs(output_dir, exist_ok=True)

# === Load data
adata_striped = sc.read(adata_striped_path)
adata_destriped = sc.read(adata_destriped_path)

# === Cluster/sample target settings
target_striped_cluster = "26"
target_destriped_cluster = "27"
samples = ["WT_A1", "WT_D1", "VSNL1_MUT_A1", "VSNL1_MUT_D1"]

# === Build count table
records = []
for sample in samples:
    striped_count = adata_striped.obs.query(
        "sample == @sample and leiden_qc == @target_striped_cluster and labels_joint_source == 'primary'"
    ).shape[0]

    destriped_count = adata_destriped.obs.query(
        "sample == @sample and leiden_qc == @target_destriped_cluster and labels_joint_source == 'primary'"
    ).shape[0]

    records.append({"sample": sample, "cluster": "Striped_26", "count": striped_count})
    records.append({"sample": sample, "cluster": "Destriped_27", "count": destriped_count})

df = pd.DataFrame(records)

# === Plot
plt.figure(figsize=(8, 6))
sns.barplot(data=df, x="sample", y="count", hue="cluster")
plt.title("Primary Cells in Cluster 26 (Striped) vs Cluster 27 (Destriped)")
plt.ylabel("Primary Cell Count")
plt.xlabel("Sample")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "barplot_primary_cluster_26_vs_27_per_sample.png"), dpi=300)
plt.close()

print("✅ Saved bar plot to:", os.path.join(output_dir, "barplot_primary_cluster_26_vs_27_per_sample.png"))

