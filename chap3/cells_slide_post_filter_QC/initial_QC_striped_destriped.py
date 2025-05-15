import os
import scanpy as sc
import pandas as pd

# === Config ===
samples = ["WT_A1", "WT_D1", "VSNL1_MUT_A1", "VSNL1_MUT_D1"]
striped_path = "summary_plots/qc_filtered_unstripped/adata_qc_filtered_clustered.h5ad"
destriped_path = "summary_plots/qc_filtered_recombined/adata_qc_filtered_clustered.h5ad"
output_csv = "qc_summary_striped_vs_destriped.csv"

# === Load objects
adata_striped = sc.read(striped_path)
adata_destriped = sc.read(destriped_path)

# === Function to summarize a dataset
def summarize(adata, sample, source_type):
    sub = adata[adata.obs["sample"] == sample].copy()
    stats = {
        "sample": sample,
        "source": source_type,
        "n_cells": sub.n_obs,
        "mean_total_counts": sub.obs["total_counts"].mean(),
        "mean_n_genes": sub.obs["n_genes_by_counts"].mean()
    }

    if "labels_joint_source" in sub.obs.columns:
        counts = sub.obs["labels_joint_source"].value_counts()
        stats["n_primary"] = counts.get("primary", 0)
        stats["n_secondary"] = counts.get("secondary", 0)
    else:
        stats["n_primary"] = "NA"
        stats["n_secondary"] = "NA"

    return stats

# === Collect all stats
rows = []
for sample in samples:
    rows.append(summarize(adata_striped, sample, "striped"))
    rows.append(summarize(adata_destriped, sample, "destriped"))

# === Save results
df = pd.DataFrame(rows)
df.to_csv(output_csv, index=False)
print(f"✅ QC summary saved to: {output_csv}")

