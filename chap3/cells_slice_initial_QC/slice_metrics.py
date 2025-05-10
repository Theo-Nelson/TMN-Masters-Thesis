import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# === Config ===
input_dir = Path("split_cells_kmeans_all")
output_dir = Path("split_cell_qc_summary")
output_dir.mkdir(exist_ok=True)
qc_summary_csv = output_dir / "per_slice_qc_summary.csv"

# === Find all *_cells.h5ad files
cell_files = sorted(input_dir.glob("*_cells.h5ad"))

# === Prepare QC summary and per-slice obs collection
qc_summary = []
all_obs = []

for path in cell_files:
    slice_name = path.stem.replace("_cells", "")
    print(f"🔍 Processing {slice_name}...")

    adata = sc.read(path)
    sc.pp.calculate_qc_metrics(adata, inplace=True)

    # Annotate
    adata.obs["slice"] = slice_name

    # Append obs for violin plots
    obs_subset = adata.obs[["total_counts", "n_genes_by_counts"]].copy()
    if "bin_count" in adata.obs:
        obs_subset["bin_count"] = adata.obs["bin_count"]
    obs_subset["slice"] = slice_name
    all_obs.append(obs_subset)

    # Summarize
    qc_summary.append({
        "slice": slice_name,
        "n_cells": adata.n_obs,
        "mean_total_counts": adata.obs["total_counts"].mean(),
        "mean_n_genes_by_counts": adata.obs["n_genes_by_counts"].mean(),
        "mean_bin_count": adata.obs["bin_count"].mean() if "bin_count" in adata.obs else None
    })

# === Write summary CSV
qc_df = pd.DataFrame(qc_summary)
qc_df.to_csv(qc_summary_csv, index=False)
print(f"✅ Saved slice summary: {qc_summary_csv}")

# === Concatenate all obs for violin plots
all_obs_df = pd.concat(all_obs)

# === Plot distributions
sns.set(style="whitegrid")
metrics = ["total_counts", "n_genes_by_counts"]
if "bin_count" in all_obs_df.columns:
    metrics.append("bin_count")

for metric in metrics:
    plt.figure(figsize=(10, 4))
    sns.violinplot(data=all_obs_df, x="slice", y=metric, inner="quartile", scale="width")
    plt.xticks(rotation=90)
    plt.title(f"{metric.replace('_', ' ').title()} per Cell (by Slice)")
    plt.tight_layout()
    plt.savefig(output_dir / f"{metric}_violinplot.png", dpi=300)
    plt.close()

print(f"✅ Violin plots saved to: {output_dir}")

