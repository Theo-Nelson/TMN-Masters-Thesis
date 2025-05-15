import os
import scanpy as sc
import pandas as pd

# === Config ===
samples = ["WT_A1", "WT_D1", "VSNL1_MUT_A1", "VSNL1_MUT_D1"]
striped_base = "bin2cell_output_unstripped"
destriped_base = "."
output_csv = "qc_summary_striped_vs_destriped_unfiltered.csv"

def summarize_h5ad(path, sample, source_label):
    adata = sc.read(path)

    stats = {
        "sample": sample,
        "source": source_label,
        "n_cells": adata.n_obs,
        "mean_total_counts": adata.obs["total_counts"].mean() if "total_counts" in adata.obs else "NA",
        "mean_n_genes_by_counts": adata.obs["n_genes_by_counts"].mean() if "n_genes_by_counts" in adata.obs else "NA"
    }

    if "labels_joint_source" in adata.obs.columns:
        counts = adata.obs["labels_joint_source"].value_counts()
        stats["n_primary"] = counts.get("primary", 0)
        stats["n_secondary"] = counts.get("secondary", 0)
    else:
        stats["n_primary"] = "NA"
        stats["n_secondary"] = "NA"

    return stats

# === Collect stats from all objects
rows = []
for sample in samples:
    # Striped object (unfiltered)
    striped_path = os.path.join(striped_base, sample, f"{sample}_cells.h5ad")
    if os.path.exists(striped_path):
        rows.append(summarize_h5ad(striped_path, sample, "striped"))
    else:
        print(f"⚠️ Skipping missing striped file for {sample}")

    # Destriped object (unfiltered)
    destriped_path = os.path.join(f"{sample}_bin2cell_output", f"{sample}_cells.h5ad")
    if os.path.exists(destriped_path):
        rows.append(summarize_h5ad(destriped_path, sample, "destriped"))
    else:
        print(f"⚠️ Skipping missing destriped file for {sample}")

# === Save summary
df = pd.DataFrame(rows)
df.to_csv(output_csv, index=False)
print(f"✅ Saved summary to: {output_csv}")

