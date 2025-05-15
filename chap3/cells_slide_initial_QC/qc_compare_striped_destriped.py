import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Config ===
samples = ["WT_A1", "WT_D1", "VSNL1_MUT_A1", "VSNL1_MUT_D1"]
striped_dir = "bin2cell_output_unstripped"
destriped_dir = "."
output_dir = "qc_summary_striped_vs_destriped"
os.makedirs(output_dir, exist_ok=True)

all_obs = []

# === Load and extract data
for sample in samples:
    for source in ["striped", "destriped"]:
        path = (
            os.path.join(striped_dir, sample, f"{sample}_cells.h5ad")
            if source == "striped"
            else os.path.join(f"{sample}_bin2cell_output", f"{sample}_cells.h5ad")
        )
        if not os.path.exists(path):
            print(f"⚠️ Missing: {source} {sample}")
            continue

        adata = sc.read(path)
        print(f"✅ Loaded {sample} ({source}): shape {adata.shape}")
        sc.pp.calculate_qc_metrics(adata, inplace=True)

        obs_df = adata.obs[["total_counts", "n_genes_by_counts", "bin_count"]].copy()
        obs_df["sample"] = sample
        obs_df["source"] = source
        obs_df["labels_joint_source"] = adata.obs.get("labels_joint_source", pd.Series(["NA"]*adata.shape[0], index=adata.obs_names))
        all_obs.append(obs_df)

# === Combine
qc_df = pd.concat(all_obs).reset_index(drop=True)

# === Violin plots per metric
metrics = ["total_counts", "n_genes_by_counts", "bin_count"]
subsets = {
    "all": qc_df,
    "primary": qc_df[qc_df["labels_joint_source"] == "primary"],
    "secondary": qc_df[qc_df["labels_joint_source"] == "secondary"]
}

sns.set(style="whitegrid")
for metric in metrics:
    for subset_name, df in subsets.items():
        plt.figure(figsize=(8, 5))
        sns.violinplot(data=df, x="sample", y=metric, hue="source", split=True, inner="quartile", scale="width")
        plt.title(f"{metric.replace('_', ' ').title()} – {subset_name} cells")
        plt.xticks(rotation=45)
        plt.tight_layout()
        filename = f"{metric}_violin_{subset_name}.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300)
        plt.close()

# === Per-sample scatterplots
for sample in samples:
    for source in ["striped", "destriped"]:
        sub = qc_df[(qc_df["sample"] == sample) & (qc_df["source"] == source)]

        if sub.empty:
            continue

        plt.figure(figsize=(6, 6))
        sns.scatterplot(data=sub, x="bin_count", y="total_counts", alpha=0.3, s=5)
        plt.title(f"{sample} ({source}): UMI vs Bin Count")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{sample}_{source}_umi_vs_bin.png"), dpi=300)
        plt.close()

        plt.figure(figsize=(6, 6))
        sns.scatterplot(data=sub, x="bin_count", y="n_genes_by_counts", alpha=0.3, s=5)
        plt.title(f"{sample} ({source}): Gene Count vs Bin Count")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{sample}_{source}_gene_vs_bin.png"), dpi=300)
        plt.close()

print(f"✅ All QC comparison plots saved to: {output_dir}")

