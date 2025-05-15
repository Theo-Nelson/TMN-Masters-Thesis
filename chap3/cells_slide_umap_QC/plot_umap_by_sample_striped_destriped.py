import scanpy as sc
import os
import matplotlib.pyplot as plt

# === Paths ===
striped_path = "summary_plots/qc_filtered_unstripped/adata_qc_filtered_clustered.h5ad"
destriped_path = "summary_plots/qc_filtered_recombined/adata_qc_filtered_clustered.h5ad"
output_dir = "qc_summary_striped_vs_destriped"
os.makedirs(output_dir, exist_ok=True)

# === Helper function to plot UMAP by sample
def plot_umap_by_sample(adata, label, out_prefix):
    if "sample" not in adata.obs:
        print(f"❌ No 'sample' column in {label}. Cannot plot.")
        return
    
    sc.pl.umap(
        adata,
        color="sample",
        title=f"UMAP Colored by Sample — {label}",
        show=False
    )
    plt.savefig(os.path.join(output_dir, f"umap_by_sample_{out_prefix}.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: umap_by_sample_{out_prefix}.png")

# === Load and plot striped
adata_striped = sc.read(striped_path)
plot_umap_by_sample(adata_striped, label="Striped", out_prefix="striped")

# === Load and plot destriped
adata_destriped = sc.read(destriped_path)
plot_umap_by_sample(adata_destriped, label="Destriped", out_prefix="destriped")

