import os
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt

# === Config ===
samples = ["WT_A1", "WT_D1", "VSNL1_MUT_A1", "VSNL1_MUT_D1"]
mpp = 0.2
img_key = f"{mpp}_mpp_150_buffer"
basis = "spatial_cropped_150_buffer"

excluded_genes = {
    "Alb", "Fabp1", "Fga", "Fgb", "Plg", "Apoc3", "C9", "Itih1", "Pigr",
    "Hamp2", "Apoa5", "Ces3a", "Vnn3", "Trf", "Apoc1", "Ahsg", "Adh1"
}

output_dir = "summary_plots/qc_filtered_recombined"
os.makedirs(output_dir, exist_ok=True)

# === Load and merge original .h5ad files using object_id ===
adata_list = []
for sample_id in samples:
    path = f"{sample_id}_bin2cell_output/{sample_id}_cells.h5ad"
    adata = sc.read(path)
    adata.obs["sample"] = sample_id
    adata.obs_names = [f"{sample_id}_{oid}" for oid in adata.obs["object_id"]]
    adata_list.append(adata)

adata = adata_list[0].concatenate(
    *adata_list[1:],
    batch_key="sample",
    batch_categories=samples,
    index_unique=None
)
print(f"✅ Merged shape: {adata.shape}")

# === QC filtering ===
sc.pp.calculate_qc_metrics(adata, inplace=True)
adata = adata[(adata.obs["n_genes_by_counts"] > 200) & (adata.obs["total_counts"] > 400)].copy()
print(f"✅ After QC filter: {adata.shape[0]} cells")

# === Exclude unwanted genes ===
genes_to_keep = [g for g in adata.var_names if g not in excluded_genes]
adata = adata[:, genes_to_keep].copy()
print(f"✅ After gene exclusion: {adata.shape[1]} genes")

# === Clustering pipeline ===
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
sc.pp.scale(adata)
sc.tl.pca(adata)
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
sc.tl.leiden(adata, key_added="leiden_qc")
sc.tl.umap(adata)

# === Save processed object
adata.write(os.path.join(output_dir, "adata_qc_filtered_clustered.h5ad"))
print("✅ Saved processed AnnData object")

# === Save UMAP plot
sc.pl.umap(adata, color="leiden_qc", title="Leiden Clusters (QC-filtered)", show=False)
plt.savefig(os.path.join(output_dir, "umap_qc_filtered.png"), dpi=300, bbox_inches="tight")
plt.close()
print("✅ Saved UMAP plot")

# === Marker genes
sc.tl.rank_genes_groups(adata, groupby="leiden_qc", method="wilcoxon")
ranked_df = sc.get.rank_genes_groups_df(adata, group=None)
marker_dir = os.path.join(output_dir, "top_markers")
os.makedirs(marker_dir, exist_ok=True)

for cluster in adata.obs["leiden_qc"].cat.categories:
    top100 = ranked_df[ranked_df["group"] == cluster].head(100)
    top100.to_csv(os.path.join(marker_dir, f"top100_markers_cluster_{cluster}.csv"), index=False)

print("✅ Saved top marker genes per cluster")

# === Prepare for spatial plotting
adata.obs["array_row_r"] = adata.obs["array_row"].round(2)
adata.obs["array_col_r"] = adata.obs["array_col"].round(2)

# === Spatial cluster plots per sample
for sample_id in samples:
    print(f"\n🗺 Plotting clusters for {sample_id}...")
    path = f"{sample_id}_bin2cell_output/{sample_id}_cells.h5ad"
    adata_sample = sc.read(path)

    # Fix spatial key
    if "spatial" in adata_sample.uns and sample_id not in adata_sample.uns["spatial"]:
        old_key = list(adata_sample.uns["spatial"].keys())[0]
        adata_sample.uns["spatial"][sample_id] = adata_sample.uns["spatial"].pop(old_key)
        print(f"🔧 Reassigned spatial key from '{old_key}' to '{sample_id}'")

    # Round coordinates for matching
    adata_sample.obs["array_row_r"] = adata_sample.obs["array_row"].round(2)
    adata_sample.obs["array_col_r"] = adata_sample.obs["array_col"].round(2)

    # Merge leiden_qc
    cluster_df = adata[adata.obs["sample"] == sample_id].obs[["array_row_r", "array_col_r", "leiden_qc"]]
    merged = pd.merge(
        adata_sample.obs.reset_index(),
        cluster_df,
        on=["array_row_r", "array_col_r"],
        how="left"
    ).drop_duplicates(subset="index", keep="first").set_index("index")

    # Assign safely
    adata_sample.obs["leiden_qc"] = merged["leiden_qc"].astype(str).fillna("NA")

    # Plot clusters
    plot_dir = os.path.join(f"{sample_id}_bin2cell_output", "spatial_qc_filtered_clusters")
    os.makedirs(plot_dir, exist_ok=True)

    for cluster in sorted(adata_sample.obs["leiden_qc"].unique()):
        if cluster == "NA":
            continue
        sub = adata_sample[adata_sample.obs["leiden_qc"] == cluster]
        if sub.n_obs == 0:
            continue
        sc.pl.spatial(
            sub,
            color="leiden_qc",
            img_key=img_key,
            basis=basis,
            title=f"{sample_id} — Cluster {cluster}",
            show=False,
            legend_loc=None
        )
        out_path = os.path.join(plot_dir, f"{sample_id}_cluster_{cluster}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

    print(f"✅ Saved spatial cluster plots to: {plot_dir}")

