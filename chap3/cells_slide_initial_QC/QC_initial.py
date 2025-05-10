import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Config ===
samples = ["WT_A1", "WT_D1", "VSNL1_MUT_A1", "VSNL1_MUT_D1"]
input_template = "{}_bin2cell_output/{}_cells.h5ad"
output_dir = "qc_summary_across_samples"
os.makedirs(output_dir, exist_ok=True)

# === Collect data
all_obs = []

for sample in samples:
    path = input_template.format(sample, sample)
    adata = sc.read(path)
    print(f"✅ Loaded {sample}: shape {adata.shape}")
    
    # Add QC metrics
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    
    # Add sample label
    adata.obs["sample"] = sample
    
    # Keep only relevant columns
    obs_df = adata.obs[["total_counts", "n_genes_by_counts", "bin_count"]].copy()
    obs_df["sample"] = sample
    
    all_obs.append(obs_df)

# === Concatenate all samples
qc_df = pd.concat(all_obs)
qc_df.reset_index(inplace=True)

# === Plot distributions
sns.set(style="whitegrid")

for metric in ["total_counts", "n_genes_by_counts", "bin_count"]:
    plt.figure(figsize=(6, 4))
    sns.violinplot(data=qc_df, x="sample", y=metric, inner="quartile", scale="width")
    plt.title(f"{metric.replace('_', ' ').title()} per Cell")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{metric}_violinplot.png"), dpi=300)
    plt.close()

# === Pairwise scatter plots per sample
for sample in samples:
    sub_df = qc_df[qc_df["sample"] == sample]
    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=sub_df, x="bin_count", y="total_counts", alpha=0.2, s=5)
    plt.title(f"{sample}: UMI vs Bin Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{sample}_umi_vs_bin_count.png"), dpi=300)
    plt.close()
    
    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=sub_df, x="bin_count", y="n_genes_by_counts", alpha=0.2, s=5)
    plt.title(f"{sample}: Gene Count vs Bin Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{sample}_gene_vs_bin_count.png"), dpi=300)
    plt.close()

print(f"✅ All QC plots saved to: {output_dir}")

