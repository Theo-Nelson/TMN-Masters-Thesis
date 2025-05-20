import os
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt

# === Config ===
input_merged = "summary_plots/qc_filtered_unstripped/adata_qc_filtered_clustered.h5ad"
samples = ["WT_A1", "WT_D1"]
base_dir = "bin2cell_output_unstripped"
mpp = 0.2
img_key = f"{mpp}_mpp_150_buffer"
basis = "spatial_cropped_150_buffer"

# === Marker genes (top 10 per cluster)
top_genes_20 = ["Actg2", "Col1a1", "Myh11", "Col3a1", "Col1a2", "Mylk", "Tpm2", "Cald1", "Acta2", "Flna"]
top_genes_26 = ["Krt15", "Krt5", "Krt19", "Igfbp5", "Ptprf", "Lrig1", "Igfbp2", "Nnat", "Trim29", "Sema3d"]

output_dir = "cluster_20_26_marker_expression"
umap_dir = os.path.join(output_dir, "umaps")
spatial_dir = os.path.join(output_dir, "spatial_maps")
spatial_expr_dir = os.path.join(output_dir, "spatial_gene_expression")

os.makedirs(umap_dir, exist_ok=True)
os.makedirs(spatial_dir, exist_ok=True)
os.makedirs(spatial_expr_dir, exist_ok=True)

# === Load clustered object
adata = sc.read(input_merged)

# === Subset to clusters 20 and 26
adata_subset = adata[adata.obs["leiden_qc"].isin(["20", "26"])].copy()

# === Plot UMAPs colored by expression
for gene in top_genes_20 + top_genes_26:
    if gene not in adata.var_names:
        continue
    sc.pl.umap(
        adata_subset,
        color=gene,
        title=f"{gene} expression (clusters 20 & 26)",
        vmin='p1', vmax='p99.5',
        show=False
    )
    plt.savefig(os.path.join(umap_dir, f"umap_cluster_20_26_{gene}.png"), dpi=300, bbox_inches="tight")
    plt.close()

# === Spatial plots per sample for clusters 20 and 26 and gene overlays
adata.obs["array_row_r"] = adata.obs["array_row"].round(2)
adata.obs["array_col_r"] = adata.obs["array_col"].round(2)

for sample_id in samples:
    print(f"\n🧬 Plotting for sample: {sample_id}")
    sample_path = f"{base_dir}/{sample_id}/{sample_id}_cells.h5ad"
    adata_sample = sc.read(sample_path)

    # Fix spatial key if needed
    if "spatial" in adata_sample.uns and sample_id not in adata_sample.uns["spatial"]:
        old_key = list(adata_sample.uns["spatial"].keys())[0]
        adata_sample.uns["spatial"][sample_id] = adata_sample.uns["spatial"].pop(old_key)

    # Round coords
    adata_sample.obs["array_row_r"] = adata_sample.obs["array_row"].round(2)
    adata_sample.obs["array_col_r"] = adata_sample.obs["array_col"].round(2)

    # Merge leiden_qc
    merged_df = adata[adata.obs["sample"] == sample_id].obs[["array_row_r", "array_col_r", "leiden_qc"]]
    merged = pd.merge(
        adata_sample.obs.reset_index(),
        merged_df,
        on=["array_row_r", "array_col_r"],
        how="left"
    ).drop_duplicates(subset="index", keep="first").set_index("index")
    adata_sample.obs["leiden_qc"] = merged["leiden_qc"].astype(str).fillna("NA")
    adata_sample.obs["leiden_qc"] = adata_sample.obs["leiden_qc"].astype("category")

    for cluster in ["20", "26"]:
        sub = adata_sample[adata_sample.obs["leiden_qc"] == cluster]
        if sub.n_obs == 0:
            continue

        # Plot spatial cluster
        sc.pl.spatial(
            sub,
            color="leiden_qc",
            img_key=img_key,
            basis=basis,
            title=f"{sample_id} — Cluster {cluster}",
            show=False,
            legend_loc=None
        )
        plt.savefig(os.path.join(spatial_dir, f"{sample_id}_cluster_{cluster}.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # Plot expression per gene for each cluster
        gene_list = top_genes_20 if cluster == "20" else top_genes_26
        for gene in gene_list:
            if gene not in adata_sample.var_names:
                continue
            sc.pl.spatial(
                sub,
                color=gene,
                img_key=img_key,
                basis=basis,
                vmin='p1',
                vmax='p99.5',
                title=f"{sample_id} — Cluster {cluster} — {gene}",
                show=False,
                legend_loc=None
            )
            plt.savefig(os.path.join(spatial_expr_dir, f"{sample_id}_cluster_{cluster}_gene_{gene}.png"),
                        dpi=300, bbox_inches="tight")
            plt.close()

# === UMAP with clusters 20 and 26 colored
sc.pl.umap(
    adata_subset,
    color="leiden_qc",
    title="UMAP: Clusters 20 and 26",
    palette=["#1f77b4", "#ff7f0e"],  # Blue & orange
    show=False
)
plt.savefig(os.path.join(umap_dir, "umap_cluster_20_26_annotated.png"), dpi=300, bbox_inches="tight")
plt.close()

# === Combined spatial overlay per sample for both clusters
for sample_id in samples:
    print(f"🧬 Combined overlay for {sample_id}")
    path = f"{base_dir}/{sample_id}/{sample_id}_cells.h5ad"
    adata_sample = sc.read(path)

    if "spatial" in adata_sample.uns and sample_id not in adata_sample.uns["spatial"]:
        old_key = list(adata_sample.uns["spatial"].keys())[0]
        adata_sample.uns["spatial"][sample_id] = adata_sample.uns["spatial"].pop(old_key)

    adata_sample.obs["array_row_r"] = adata_sample.obs["array_row"].round(2)
    adata_sample.obs["array_col_r"] = adata_sample.obs["array_col"].round(2)

    merged_df = adata[adata.obs["sample"] == sample_id].obs[["array_row_r", "array_col_r", "leiden_qc"]]
    merged = pd.merge(
        adata_sample.obs.reset_index(),
        merged_df,
        on=["array_row_r", "array_col_r"],
        how="left"
    ).drop_duplicates(subset="index", keep="first").set_index("index")
    adata_sample.obs["leiden_qc"] = merged["leiden_qc"].astype(str).fillna("NA")
    adata_sample.obs["leiden_qc"] = adata_sample.obs["leiden_qc"].astype("category")

    subset_combined = adata_sample[adata_sample.obs["leiden_qc"].isin(["20", "26"])]
    if subset_combined.n_obs > 0:
        sc.pl.spatial(
            subset_combined,
            color="leiden_qc",
            img_key=img_key,
            basis=basis,
            title=f"{sample_id} — Clusters 20 & 26",
            show=False,
            palette=["#1f77b4", "#ff7f0e"]
        )
        plt.savefig(os.path.join(spatial_dir, f"{sample_id}_cluster_20_26_combined.png"), dpi=300, bbox_inches="tight")
        plt.close()


print("\n✅ All UMAP and spatial plots for clusters 20 and 26 saved.")

