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

variants = {
    "striped": {
        "input_base": "bin2cell_output_unstripped",
        "output_dir": "summary_plots/qc_filtered_unstripped_primary_only"
    },
    "destriped": {
        "input_base": "",  # cells live in <sample>_bin2cell_output/
        "output_dir": "summary_plots/qc_filtered_recombined_primary_only"
    }
}

for label, config in variants.items():
    print(f"\n🚀 Processing {label.upper()} (primary-only)...")

    adata_list = []
    for sample_id in samples:
        if label == "striped":
            path = f"{config['input_base']}/{sample_id}/{sample_id}_cells.h5ad"
        else:
            path = f"{sample_id}_bin2cell_output/{sample_id}_cells.h5ad"

        if not os.path.exists(path):
            print(f"⚠️ Skipping missing file: {path}")
            continue

        adata = sc.read(path)
        adata.obs["sample"] = sample_id
        adata.obs_names = [f"{sample_id}_{oid}" for oid in adata.obs["object_id"]]

        # Filter primary only
        if "labels_joint_source" in adata.obs:
            adata = adata[adata.obs["labels_joint_source"] == "primary"].copy()
            print(f"✅ {sample_id}: {adata.shape[0]} primary cells retained")

        adata_list.append(adata)

    # Merge
    adata = adata_list[0].concatenate(
        *adata_list[1:],
        batch_key="sample",
        batch_categories=samples,
        index_unique=None
    )
    print(f"🧬 Combined shape after merging primary cells: {adata.shape}")

    # === QC filtering
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    adata = adata[(adata.obs["n_genes_by_counts"] > 200) & (adata.obs["total_counts"] > 400)].copy()
    print(f"✅ After QC filter: {adata.shape[0]} cells")

    # === Gene exclusion
    genes_to_keep = [g for g in adata.var_names if g not in excluded_genes]
    adata = adata[:, genes_to_keep].copy()
    print(f"✅ After gene exclusion: {adata.shape[1]} genes")

    # === Clustering pipeline
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
    sc.pp.scale(adata)
    sc.tl.pca(adata)
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
    sc.tl.leiden(adata, key_added="leiden_qc")
    sc.tl.umap(adata)

    # === Save outputs
    os.makedirs(config["output_dir"], exist_ok=True)
    adata.write(os.path.join(config["output_dir"], "adata_qc_filtered_clustered.h5ad"))

    sc.pl.umap(
        adata,
        color="leiden_qc",
        title=f"Leiden Clusters (Primary-only, {label})",
        show=False
    )
    plt.savefig(os.path.join(config["output_dir"], "umap_qc_filtered.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ UMAP + clustered AnnData saved to {config['output_dir']}")

    # === Save top marker genes
    sc.tl.rank_genes_groups(adata, groupby="leiden_qc", method="wilcoxon")
    ranked_df = sc.get.rank_genes_groups_df(adata, group=None)
    marker_dir = os.path.join(config["output_dir"], "top_markers")
    os.makedirs(marker_dir, exist_ok=True)

    for cluster in adata.obs["leiden_qc"].cat.categories:
        top100 = ranked_df[ranked_df["group"] == cluster].head(100)
        top100.to_csv(os.path.join(marker_dir, f"top100_markers_cluster_{cluster}.csv"), index=False)

    print("✅ Saved marker gene tables.")

