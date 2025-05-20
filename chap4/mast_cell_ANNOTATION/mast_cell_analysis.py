# WT & VSNL1 cluster + gene expression visualization for histamine-related genes
import os
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# === Config ===
data_root = "WT_COM_centered_alignment_cells_unstripped"
data_root_vsnl = "VSNL1_COM_centered_alignment_cells_unstripped"
output_dir = "mast_cell_dotplots"
os.makedirs(output_dir, exist_ok=True)

wt_slices = [
    "WT_D1_heart_34",
    "WT_D1_heart_35",
    "WT_D1_heart_36",
    "WT_D1_heart_37",
    "WT_A1_heart_38",
    "WT_A1_heart_47",
    "WT_A1_heart_49",
    "WT_A1_heart_50",
]
vsnl_slices = [
    "VSNL1_MUT_A1_heart_19",
    "VSNL1_MUT_A1_heart_20",
    "VSNL1_MUT_A1_heart_22",
    "VSNL1_MUT_A1_heart_23",
    "VSNL1_MUT_D1_heart_24",
    "VSNL1_MUT_D1_heart_28",
    "VSNL1_MUT_D1_heart_30",
    "VSNL1_MUT_D1_heart_31",
]

highlight_cluster = "24"
gene_sets = {
    "mast_markers": ["Cma1", "Cpa3", "Mcpt2"],
    "histamine_biosynthesis": ["Hdc"],
    "granule_storage": ["Cma1", "Cpa3", "Mcpt1", "Mcpt2", "Mcpt4", "Tpsab1", "Tpsb2"],
    "packaging": ["Slc18a2"],
    "receptors": ["Hrh1", "Hrh2", "Hrh3", "Hrh4"]
}
additional_genes = [
    "Tpsg1", "Tpsb2", "Srgn", "Il1rl1", "Rab44", "Fcer1g"
]
all_genes = list(set(sum(gene_sets.values(), []) + additional_genes))


def plot_clusters(adata, slide_id, output_base):
    coords = adata.obsm["spatial_3D"][:, :2]
    cluster_labels = adata.obs["leiden_qc"].astype(str)
    highlight = cluster_labels == highlight_cluster

    plt.figure(figsize=(6, 6))
    plt.scatter(coords[~highlight, 0], coords[~highlight, 1], s=1, c="lightgrey", label="Other")
    plt.scatter(coords[highlight, 0], coords[highlight, 1], s=1.5, c="red", label=f"Cluster {highlight_cluster}")
    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.title(f"{slide_id}")
    plt.xticks([])
    plt.yticks([])
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_base, f"{slide_id}_cluster_{highlight_cluster}.png"), dpi=300)
    plt.close()


def plot_gene(adata, gene, slide_id, output_base):
    if gene not in adata.var_names:
        print(f"⚠️ {gene} not in {slide_id}")
        return

    coords = adata.obsm["spatial_3D"][:, :2]
    cluster_labels = adata.obs["leiden_qc"].astype(str)
    expr = adata[:, gene].X.toarray().flatten() if hasattr(adata[:, gene].X, "toarray") else adata[:, gene].X.flatten()

    vmin = np.percentile(expr, 0.5)
    vmax = np.percentile(expr, 99.5)

    plt.figure(figsize=(6, 6))
    sc = plt.scatter(coords[:, 0], coords[:, 1], c=expr, cmap="viridis", s=1.5, vmin=vmin, vmax=vmax)

    # Add red circle outlines for cluster 24
    highlight = cluster_labels == highlight_cluster
    if highlight.any():
        plt.scatter(coords[highlight, 0], coords[highlight, 1], s=15, facecolors='none', edgecolors='red', linewidths=0.05)

    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.title(f"{slide_id} — {gene}")
    plt.colorbar(sc, label="Expression")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(output_base, f"{slide_id}_gene_{gene}.png"), dpi=300)
    plt.close()

# === Process WT and VSNL1
for label, slices, root in [("WT", wt_slices, data_root), ("VSNL1", vsnl_slices, data_root_vsnl)]:
    out_base = os.path.join(output_dir, label)
    os.makedirs(out_base, exist_ok=True)

    for sid in slices:
        ad = sc.read(os.path.join(root, f"{sid}_cells_3D_aligned.h5ad"))
        plot_clusters(ad, sid, out_base)
        for gene in all_genes:
            plot_gene(ad, gene, sid, out_base)

print("✅ Finished cluster and expression dot plots for WT and VSNL1")
